#!/usr/bin/env python3
"""Fail-closed analysis for the fixed-feature SMT-v3 factorial.

The corresponding runner fixes a 5 environment x 9 design x 5 seed grid.  This
script deliberately refuses to summarize a partial or protocol-mismatched grid.
It first reuses the common-target analyzer, then performs the SMT-v3-specific
gate audit and calibrated-mean intervention, and finally reuses the shifted-mask
evaluator for every design.

The architecture decision thresholds are prospective internal screening constants set before the
225-cell grid was launched; they are not an external preregistration.  They are written verbatim to
``decision.json``. Aggregation code was finalized during execution and is provenance-hashed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.data import PrecomputedFeatureDataset  # noqa: E402
from lewm.models.memory_model import MemoryLeWorldModel  # noqa: E402


OCC_TO_CLEAN = {
    "dmc:reacher.hard.occ": "dmc:reacher.hard",
    "dmc:ball_in_cup.catch.occ": "dmc:ball_in_cup.catch",
    "dmc:finger.spin.occ": "dmc:finger.spin",
    "dmc:cheetah.run.occ": "dmc:cheetah.run",
    "ogbench:cube-single.occ": "ogbench:cube-single",
}
DESIGNS = (
    "none",
    "multi",
    "gru",
    "ssm",
    "smt",
    "smtv3_static",
    "smtv3",
    "smtv3_old",
    "smtv3_oracle",
)
V3_DESIGNS = ("smtv3_static", "smtv3", "smtv3_old", "smtv3_oracle")
SEEDS = (0, 1, 2, 3, 4)
DEFAULT_EPOCHS = 200
CONVERGENCE_WINDOW = 10
FIRST_POST_LOSS_WEIGHT = 0.5
SHIFTED_CONDITIONS = ("early_6_12", "late_14_20", "longer_10_19")

# Predeclared architecture-audit decision thresholds.
GO_MIN_ENV_REL10 = 4
GO_MIN_PAIRED_WINS = 20
GO_MIN_HOLD_ENV_WINS = 4
GO_MIN_ERASURE_FRACTION = 0.50
GO_MIN_GATE_AUROC = 0.90
GO_MIN_GATE_GAP = 0.20
GO_MAX_CLEAN_INPUT_WORSENING = 0.05
ORACLE_MIN_ENV_WINS = 4
NO_GO_MAX_DYNAMIC_STATIC_GAIN = 0.05
NO_GO_MAX_MEAN_INTERVENTION_CHANGE = 0.03
CONVERGENCE_MEDIAN_MAX = 0.01
CONVERGENCE_CELL_MAX = 0.03

EXPECTED = set(itertools.product(OCC_TO_CLEAN, DESIGNS, SEEDS))
NUM_RUNS = len(EXPECTED)
NUM_GATE_RUNS = len(OCC_TO_CLEAN) * len(V3_DESIGNS) * len(SEEDS)
if NUM_RUNS != 225:  # guard accidental edits to the protocol constants
    raise RuntimeError(f"internal protocol error: expected 225 cells, got {NUM_RUNS}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the exact 225-cell fixed-DINO SMT-v3 factorial.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", type=Path, default=Path("outputs/smt_v3_shared"))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    args = parser.parse_args(argv)
    args.root = args.root.resolve()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    if args.epochs < 2 * CONVERGENCE_WINDOW:
        parser.error(
            f"--epochs must be at least {2 * CONVERGENCE_WINDOW} so two disjoint "
            f"{CONVERGENCE_WINDOW}-epoch descriptive windows exist"
        )
    return args


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is unavailable")
    return device


def safe_env_name(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_")


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject_non_rfc_json(value: str) -> None:
    raise ValueError(f"non-RFC JSON constant {value}")


def semantically_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(
            semantically_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            semantically_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, float) and isinstance(right, float):
        if math.isnan(left) or math.isnan(right):
            return math.isnan(left) and math.isnan(right)
    return left == right


def finite(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"non-finite {label}: {value!r}")
    return float(value)


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return float(sum(values) / len(values))


def popstd(values: Sequence[float]) -> float:
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / len(values))


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError(f"inconsistent CSV row schema for {path}")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    atomic_write_text(path, text)


def read_csv_strict(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing/nonempty prerequisite CSV: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"CSV has no rows: {path}")
    return rows


def run_dependency(command: Sequence[str]) -> None:
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def expected_run_name(env: str, design: str, seed: int) -> str:
    return f"lewm-{env}-{design}-s{seed}"


def validate_artifact_layout(root: Path) -> dict[tuple[str, str, int], tuple[Path, Path]]:
    if not root.is_dir():
        raise FileNotFoundError(f"experiment root does not exist: {root}")
    expected = {
        (env, design, seed): root / expected_run_name(env, design, seed)
        for env, design, seed in EXPECTED
    }
    expected_dirs = {path.name for path in expected.values()}
    actual_dirs = {path.name for path in root.glob("lewm-*") if path.is_dir()}
    if actual_dirs != expected_dirs:
        raise ValueError(
            "run-directory set is not the exact 225-cell factorial; "
            f"missing={sorted(expected_dirs - actual_dirs)[:8]}, "
            f"extra={sorted(actual_dirs - expected_dirs)[:8]}"
        )
    expected_models = {path / "model.pt" for path in expected.values()}
    expected_metrics = {path / "metrics.json" for path in expected.values()}
    actual_models = set(root.rglob("model.pt"))
    actual_metrics = set(root.rglob("metrics.json"))
    if actual_models != expected_models or actual_metrics != expected_metrics:
        raise ValueError(
            "checkpoint/metrics set is not exact; "
            f"missing_models={sorted(map(str, expected_models - actual_models))[:4]}, "
            f"extra_models={sorted(map(str, actual_models - expected_models))[:4]}, "
            f"missing_metrics={sorted(map(str, expected_metrics - actual_metrics))[:4]}, "
            f"extra_metrics={sorted(map(str, actual_metrics - expected_metrics))[:4]}"
        )
    for artifact in expected_models | expected_metrics:
        if not artifact.is_file() or artifact.stat().st_size == 0:
            raise ValueError(f"missing, empty, or non-file run artifact: {artifact}")
    return {
        key: (run_dir / "model.pt", run_dir / "metrics.json")
        for key, run_dir in expected.items()
    }


def invoke_common_analysis(root: Path, epochs: int) -> None:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "analyze_shared_clean_occlusion.py"),
        "--root",
        str(root),
        "--seeds",
        *map(str, SEEDS),
        "--designs",
        *DESIGNS,
        "--num-episodes",
        "600",
        "--val-episodes",
        "150",
        "--length",
        "32",
        "--epochs",
        str(epochs),
        "--feature-dim",
        "128",
    ]
    run_dependency(command)
    rows = read_csv_strict(root / "per_run.csv")
    found = {(row["env"], row["design"], int(row["seed"])) for row in rows}
    if len(rows) != NUM_RUNS or found != EXPECTED:
        raise ValueError("common-target analyzer did not attest the exact 225 cells")


def exact_config(
    root: Path, env: str, design: str, seed: int, epochs: int
) -> dict[str, Any]:
    return {
        "env_id": env,
        "target_env_id": OCC_TO_CLEAN[env],
        "memory_mode": design,
        "smt_router": "sigmoid",
        "seed": seed,
        "num_episodes": 600,
        "val_episodes": 150,
        "prototype_seed": 0,
        "mask_occluded_target_loss": True,
        "first_post_loss_weight": FIRST_POST_LOSS_WEIGHT,
        "freeze_encoder": False,
        "encoder_type": "precomputed",
        "length": 32,
        "img_size": 64,
        "epochs": epochs,
        "batch_size": 64,
        "lr": 3e-4,
        "weight_decay": 1e-5,
        "num_workers": 2,
        "no_amp": False,
        "patch_size": 8,
        "embed_dim": 128,
        "encoder_layers": 6,
        "encoder_heads": 4,
        "predictor_layers": 4,
        "predictor_heads": 8,
        "history_len": 3,
        "dropout": 0.1,
        "sigreg_lambda": 0.1,
        "sigreg_projections": 512,
        "tau_fast": 3.0,
        "tau_slow": 25.0,
        "fixed_alpha": True,
        "wandb": False,
        "device": "cuda",
    }


def validate_history(run: str, history: Any, epochs: int) -> dict[str, Any]:
    if not isinstance(history, list) or len(history) != epochs:
        raise ValueError(
            f"{run}: history length={len(history) if isinstance(history, list) else None}, "
            f"expected {epochs}"
        )
    values: dict[int, tuple[float, float]] = {}
    for expected_epoch, record in enumerate(history, 1):
        if not isinstance(record, Mapping) or record.get("epoch") != expected_epoch:
            raise ValueError(f"{run}: malformed history at epoch {expected_epoch}")
        for split in ("train", "val"):
            metrics = record.get(split)
            if not isinstance(metrics, Mapping):
                raise ValueError(f"{run}: missing {split} history at epoch {expected_epoch}")
            for key in ("loss", "pred_loss", "sigreg_loss"):
                finite(metrics.get(key), f"{run} epoch {expected_epoch} {split}.{key}")
        values[expected_epoch] = (
            float(record["train"]["pred_loss"]),
            float(record["val"]["pred_loss"]),
        )
    val_values = [values[epoch][1] for epoch in range(1, epochs + 1)]
    best_index = int(np.argmin(np.asarray(val_values)))
    midpoint = epochs // 2
    window_start = epochs - CONVERGENCE_WINDOW
    rel_improvement = (
        values[window_start][1] - values[epochs][1]
    ) / values[window_start][1]

    # Noise-robust diagnostics are deliberately descriptive and do not replace the locked
    # point-to-point convergence rule above.  At the default 200 epochs these compare the
    # inclusive windows 181--190 and 191--200.  The slope is ordinary least squares over the
    # recent window, expressed as objective change per epoch and normalized by its window mean.
    descriptive_previous_start = epochs - 2 * CONVERGENCE_WINDOW + 1
    descriptive_previous_end = epochs - CONVERGENCE_WINDOW
    descriptive_recent_start = descriptive_previous_end + 1
    previous_values = np.asarray(
        [
            values[epoch][1]
            for epoch in range(descriptive_previous_start, descriptive_previous_end + 1)
        ],
        dtype=np.float64,
    )
    recent_values = np.asarray(
        [values[epoch][1] for epoch in range(descriptive_recent_start, epochs + 1)],
        dtype=np.float64,
    )
    if len(previous_values) != CONVERGENCE_WINDOW or len(recent_values) != CONVERGENCE_WINDOW:
        raise ValueError(f"{run}: malformed descriptive convergence windows")
    previous_mean = float(previous_values.mean())
    recent_mean = float(recent_values.mean())
    descriptive_relative_improvement = (previous_mean - recent_mean) / previous_mean
    x = np.arange(CONVERGENCE_WINDOW, dtype=np.float64)
    centered_x = x - x.mean()
    recent_slope = float(
        np.dot(centered_x, recent_values - recent_mean) / np.dot(centered_x, centered_x)
    )
    normalized_recent_slope = recent_slope / recent_mean
    return {
        "epochs": epochs,
        "convergence_window_start": window_start,
        "train_pred_epoch_1": values[1][0],
        "train_pred_epoch_midpoint": values[midpoint][0],
        "train_pred_window_start": values[window_start][0],
        "train_pred_final": values[epochs][0],
        "val_pred_epoch_1": values[1][1],
        "val_pred_epoch_midpoint": values[midpoint][1],
        "val_pred_window_start": values[window_start][1],
        "val_pred_final": values[epochs][1],
        "best_val_pred": val_values[best_index],
        "best_val_epoch": best_index + 1,
        "relative_val_improvement_final_window": rel_improvement,
        "descriptive_previous_window_start_epoch": descriptive_previous_start,
        "descriptive_previous_window_end_epoch": descriptive_previous_end,
        "descriptive_recent_window_start_epoch": descriptive_recent_start,
        "descriptive_recent_window_end_epoch": epochs,
        "descriptive_val_pred_previous_window_mean": previous_mean,
        "descriptive_val_pred_recent_window_mean": recent_mean,
        "descriptive_relative_val_mean_improvement": descriptive_relative_improvement,
        "descriptive_recent_val_slope_per_epoch": recent_slope,
        "descriptive_recent_val_slope_normalized_by_mean": normalized_recent_slope,
    }


def validate_checkpoints(
    root: Path,
    artifacts: Mapping[tuple[str, str, int], tuple[Path, Path]],
    epochs: int,
) -> tuple[
    dict[tuple[str, str, int], dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, str]],
]:
    metadata: dict[tuple[str, str, int], dict[str, Any]] = {}
    convergence: list[dict[str, Any]] = []
    input_hashes: dict[str, dict[str, str]] = {}
    feature_root = (root / "dino_features_d128").resolve()
    for env, design, seed in sorted(EXPECTED):
        key = (env, design, seed)
        model_path, metrics_path = artifacts[key]
        try:
            metrics = json.loads(
                metrics_path.read_text(), parse_constant=reject_non_rfc_json
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"{metrics_path}: invalid RFC JSON: {exc}") from exc
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise ValueError(f"{model_path}: checkpoint is not a mapping")
        cfg = checkpoint.get("args")
        state = checkpoint.get("model_state_dict")
        if not isinstance(cfg, Mapping) or not isinstance(state, Mapping) or not state:
            raise ValueError(f"{model_path}: malformed args/model state")
        if not semantically_equal(metrics, checkpoint.get("final_metrics")):
            raise ValueError(f"{model_path}: metrics.json differs from final_metrics")
        for name, wanted in exact_config(root, env, design, seed, epochs).items():
            if cfg.get(name) != wanted:
                raise ValueError(
                    f"{model_path.parent.name}: {name}={cfg.get(name)!r}, expected {wanted!r}"
                )
        if cfg.get("encoder_checkpoint") is not None or cfg.get("encoder_stats") is not None:
            raise ValueError(f"{model_path.parent.name}: unexpected encoder source")
        if resolve_repo_path(cfg.get("output_dir", "")) != root:
            raise ValueError(f"{model_path.parent.name}: output_dir does not resolve to {root}")

        safe = safe_env_name(OCC_TO_CLEAN[env])
        expected_paths = {
            "feature_manifest": feature_root / f"{safe}_manifest.json",
            "train_feature_cache": feature_root / f"{safe}_train.npz",
            "val_feature_cache": feature_root / f"{safe}_val.npz",
        }
        for name, wanted in expected_paths.items():
            if resolve_repo_path(cfg.get(name, "")) != wanted or not wanted.is_file():
                raise ValueError(
                    f"{model_path.parent.name}: invalid/missing {name}: {cfg.get(name)!r}"
                )
        manifest_hash = sha256_file(expected_paths["feature_manifest"])
        if cfg.get("feature_manifest_sha256") != manifest_hash:
            raise ValueError(f"{model_path.parent.name}: feature manifest hash mismatch")
        required_metrics = (
            "val_pred_loss",
            "clean_mse_first_post",
            "last_visible_mse_first_post",
            "clean_input_mse_first_post",
        )
        for name in required_metrics:
            finite(metrics.get(name), f"{model_path.parent.name} metric {name}")
        metric_identity = {
            "env": env,
            "design": design,
            "target_env": OCC_TO_CLEAN[env],
            "external_features_fixed": True,
            "masked_clean_blackout_loss": True,
            "first_post_loss_weight": FIRST_POST_LOSS_WEIGHT,
            "primary_common_target_metric": "clean_mse_first_post",
        }
        for name, wanted in metric_identity.items():
            if metrics.get(name) != wanted:
                raise ValueError(
                    f"{model_path.parent.name}: metric {name}={metrics.get(name)!r}, "
                    f"expected {wanted!r}"
                )

        conv = validate_history(model_path.parent.name, checkpoint.get("history"), epochs)
        convergence.append(
            {
                "run": model_path.parent.name,
                "env": env,
                "design": design,
                "seed": seed,
                **conv,
            }
        )
        metadata[key] = {
            "run": model_path.parent.name,
            "model_path": model_path,
            "metrics_path": metrics_path,
            "cfg": dict(cfg),
            "metrics": dict(metrics),
            "feature_paths": expected_paths,
        }
        input_hashes[model_path.parent.name] = {
            "model.pt": sha256_file(model_path),
            "metrics.json": sha256_file(metrics_path),
        }
        del checkpoint, state

    convergence.sort(key=lambda row: (row["env"], row["design"], row["seed"]))
    if len(metadata) != NUM_RUNS:
        raise ValueError(f"validated {len(metadata)}/{NUM_RUNS} checkpoint cells")
    return metadata, convergence, input_hashes


def build_model(cfg: Mapping[str, Any], action_dim: int) -> MemoryLeWorldModel:
    mode = str(cfg["memory_mode"])
    impl = mode if mode in (
        "multi",
        "gru",
        "ssm",
        "retrieval",
        "smt",
        "smtv3",
        "smtv3_static",
        "smtv3_old",
        "smtv3_oracle",
    ) else "ema"
    ema_mode = "both" if impl != "ema" else mode
    return MemoryLeWorldModel(
        img_size=cfg["img_size"],
        patch_size=cfg["patch_size"],
        embed_dim=cfg["embed_dim"],
        action_dim=action_dim,
        encoder_layers=cfg["encoder_layers"],
        encoder_heads=cfg["encoder_heads"],
        predictor_layers=cfg["predictor_layers"],
        predictor_heads=cfg["predictor_heads"],
        history_len=cfg["history_len"],
        dropout=cfg["dropout"],
        sigreg_lambda=cfg["sigreg_lambda"],
        sigreg_projections=cfg["sigreg_projections"],
        memory_mode=ema_mode,
        memory_impl=impl,
        tau_fast=cfg["tau_fast"],
        tau_slow=cfg["tau_slow"],
        learnable_alpha=not cfg.get("fixed_alpha", True),
        smt_router=cfg.get("smt_router", "softmax"),
        encoder_type="precomputed",
    )


def make_loader(
    dataset: PrecomputedFeatureDataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


@torch.no_grad()
def collect_gates(
    model: MemoryLeWorldModel,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    chunks = []
    for observations, _actions, _targets, update_mask in loader:
        observations = observations.to(device, non_blocking=True)
        update_mask = update_mask.to(device, non_blocking=True)
        gates = model.mem_smtv3.gate_values(
            observations, memory_update_mask=update_mask
        )
        chunks.append(gates[..., 0].float().cpu().numpy())
    if not chunks:
        raise ValueError("empty feature loader while collecting gates")
    values = np.concatenate(chunks, axis=0)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("invalid SMT-v3 gates")
    return values


def binary_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUROC with exact average ranks for tied gate values."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.bool_).reshape(-1)
    if scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError("invalid AUROC scores/labels")
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC requires both classes")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    positive_rank_sum = 0.0
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0  # one-indexed ranks start+1 .. end
        positive_rank_sum += average_rank * int(sorted_labels[start:end].sum())
        start = end
    auc = (positive_rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )
    if not 0.0 <= auc <= 1.0:
        raise ValueError(f"computed invalid AUROC {auc}")
    return float(auc)


@torch.no_grad()
def evaluate_first_post(
    model: MemoryLeWorldModel,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    mean_gate: float,
) -> tuple[list[dict[str, float]], np.ndarray]:
    model.eval()
    episode_rows: list[dict[str, float]] = []
    gate_chunks = []
    episode_index = 0
    h = int(model.history_len)
    for observations, actions, targets, update_mask in loader:
        observations = observations.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        update_mask = update_mask.to(device, non_blocking=True)
        gates = model.mem_smtv3.gate_values(
            observations, memory_update_mask=update_mask
        )
        gate_chunks.append(gates[..., 0].float().cpu().numpy())
        context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if use_amp
            else nullcontext()
        )
        with context:
            injected = model._inject(
                observations, memory_update_mask=update_mask
            )
            injected_mean = model._inject(
                observations,
                memory_update_mask=update_mask,
                gate_override=mean_gate,
            )
            batch, length, dim = observations.shape
            windows = length - h
            full_windows = (
                injected.unfold(1, h, 1)[:, :windows]
                .permute(0, 1, 3, 2)
                .reshape(batch * windows, h, dim)
            )
            mean_windows = (
                injected_mean.unfold(1, h, 1)[:, :windows]
                .permute(0, 1, 3, 2)
                .reshape(batch * windows, h, dim)
            )
            action_windows = (
                actions.unfold(1, h, 1)[:, :windows]
                .permute(0, 1, 3, 2)
                .reshape(batch * windows, h, -1)
            )
            prediction = model.predictor(full_windows, action_windows)[:, -1].reshape(
                batch, windows, dim
            )
            prediction_mean = model.predictor(mean_windows, action_windows)[:, -1].reshape(
                batch, windows, dim
            )

        mask_np = update_mask[0].cpu().numpy().astype(bool)
        if not np.all(update_mask.cpu().numpy() == mask_np[None, :]):
            raise ValueError("target/update mask differs across episodes")
        hidden = np.flatnonzero(~mask_np)
        if not len(hidden) or not np.array_equal(hidden, np.arange(hidden[0], hidden[-1] + 1)):
            raise ValueError("update mask does not contain one contiguous blackout")
        start, end = int(hidden[0]), int(hidden[-1] + 1)
        first_window = end - h
        if first_window < 0 or first_window >= windows:
            raise ValueError("first-post target is outside prediction windows")
        target = targets[:, end]
        original_error = (
            (prediction[:, first_window] - target).float().square().mean(-1).cpu().numpy()
        )
        mean_error = (
            (prediction_mean[:, first_window] - target)
            .float()
            .square()
            .mean(-1)
            .cpu()
            .numpy()
        )
        hold_error = (
            (observations[:, start - 1] - target).float().square().mean(-1).cpu().numpy()
        )
        batch_gates = gate_chunks[-1]
        causal_mask = mask_np[1:end]
        if not causal_mask.any() or causal_mask.all():
            raise ValueError("causal gate prefix must contain visible and blackout timesteps")
        for local in range(batch):
            causal_gates = batch_gates[local, 1:end]
            episode_rows.append(
                {
                    "episode": episode_index,
                    "first_post_mse": float(original_error[local]),
                    "first_post_mse_mean_gate": float(mean_error[local]),
                    "delta_mse_mean_gate": float(mean_error[local] - original_error[local]),
                    "last_visible_hold_mse": float(hold_error[local]),
                    "episode_causal_gate_mean": float(causal_gates.mean()),
                    "episode_causal_gate_visible_mean": float(
                        causal_gates[causal_mask].mean()
                    ),
                    "episode_causal_gate_black_mean": float(
                        causal_gates[~causal_mask].mean()
                    ),
                    "episode_descriptive_full_sequence_gate_mean": float(
                        batch_gates[local].mean()
                    ),
                    "episode_descriptive_full_sequence_gate_visible_mean": float(
                        batch_gates[local, mask_np].mean()
                    ),
                    "episode_descriptive_full_sequence_gate_black_mean": float(
                        batch_gates[local, ~mask_np].mean()
                    ),
                }
            )
            episode_index += 1
    if not episode_rows:
        raise ValueError("empty validation feature loader")
    return episode_rows, np.concatenate(gate_chunks, axis=0)


def summarize_gate_run(
    run: str,
    env: str,
    design: str,
    seed: int,
    calibration_gates: np.ndarray,
    validation_gates: np.ndarray,
    valid_mask: np.ndarray,
    episode_rows: Sequence[Mapping[str, float]],
    reference_mse: float,
    clean_input_mse: float,
    calibration_start: int,
    calibration_end_exclusive: int,
) -> dict[str, Any]:
    if validation_gates.shape[1] != len(valid_mask):
        raise ValueError(f"{run}: gate/mask sequence length mismatch")
    if not 0 <= calibration_start < calibration_end_exclusive <= len(valid_mask):
        raise ValueError(f"{run}: invalid causal gate interval")
    if calibration_gates.ndim != 2 or calibration_gates.shape[1] != (
        calibration_end_exclusive - calibration_start
    ):
        raise ValueError(f"{run}: calibration gates do not match causal interval")
    causal_gates = validation_gates[:, calibration_start:calibration_end_exclusive]
    causal_mask = valid_mask[calibration_start:calibration_end_exclusive]
    causal_labels = np.broadcast_to(causal_mask[None, :], causal_gates.shape)
    full_labels = np.broadcast_to(valid_mask[None, :], validation_gates.shape)
    if not causal_labels.any() or causal_labels.all():
        raise ValueError(f"{run}: causal gate interval requires both label classes")
    original = [float(row["first_post_mse"]) for row in episode_rows]
    overridden = [float(row["first_post_mse_mean_gate"]) for row in episode_rows]
    hold = [float(row["last_visible_hold_mse"]) for row in episode_rows]
    result = {
        "run": run,
        "env": env,
        "design": design,
        "seed": seed,
        "n_calibration_episodes": int(calibration_gates.shape[0]),
        "n_validation_episodes": int(validation_gates.shape[0]),
        "calibration_gate_start": calibration_start,
        "calibration_gate_end_exclusive": calibration_end_exclusive,
        "decision_gate_start": calibration_start,
        "decision_gate_end_exclusive": calibration_end_exclusive,
        "calibration_gate_mean": float(calibration_gates.mean()),
        "calibration_gate_std": float(calibration_gates.std()),
        "val_causal_gate_mean": float(causal_gates.mean()),
        "val_causal_gate_visible_mean": float(causal_gates[causal_labels].mean()),
        "val_causal_gate_black_mean": float(causal_gates[~causal_labels].mean()),
        "val_causal_gate_visible_minus_black": float(
            causal_gates[causal_labels].mean() - causal_gates[~causal_labels].mean()
        ),
        "val_causal_gate_visible_auroc": binary_auroc(causal_gates, causal_labels),
        # Temporal std measures variation of the episode-mean gate across time.  Input std
        # measures episode-to-episode variation at a matched time, averaged over time.
        "val_causal_gate_temporal_std": float(causal_gates.mean(axis=0).std()),
        "val_causal_gate_input_std": float(causal_gates.std(axis=0).mean()),
        "val_causal_gate_total_std": float(causal_gates.std()),
        # Full-sequence gate summaries are retained for diagnostics only.  Gates at t=0 and
        # t>=first_post cannot affect the primary first-post prediction and never feed a decision.
        "descriptive_val_full_sequence_gate_mean": float(validation_gates.mean()),
        "descriptive_val_full_sequence_gate_visible_mean": float(
            validation_gates[full_labels].mean()
        ),
        "descriptive_val_full_sequence_gate_black_mean": float(
            validation_gates[~full_labels].mean()
        ),
        "descriptive_val_full_sequence_gate_visible_minus_black": float(
            validation_gates[full_labels].mean()
            - validation_gates[~full_labels].mean()
        ),
        "descriptive_val_full_sequence_gate_visible_auroc": binary_auroc(
            validation_gates, full_labels
        ),
        "descriptive_val_full_sequence_gate_temporal_std": float(
            validation_gates.mean(axis=0).std()
        ),
        "descriptive_val_full_sequence_gate_input_std": float(
            validation_gates.std(axis=0).mean()
        ),
        "descriptive_val_full_sequence_gate_total_std": float(
            validation_gates.std()
        ),
        "first_post_mse": mean(original),
        "first_post_mse_mean_gate": mean(overridden),
        "mean_gate_delta_mse": mean(overridden) - mean(original),
        "mean_gate_relative_change": (mean(overridden) - mean(original)) / mean(original),
        "last_visible_hold_mse": mean(hold),
        "clean_input_first_post_mse": clean_input_mse,
        "checkpoint_first_post_mse": reference_mse,
        "checkpoint_parity_abs": abs(mean(original) - reference_mse),
    }
    if result["checkpoint_parity_abs"] > 2e-4:
        raise ValueError(
            f"{run}: fresh first-post MSE {result['first_post_mse']} does not reproduce "
            f"checkpoint {reference_mse}"
        )
    return result


def evaluate_v3_gates(
    metadata: Mapping[tuple[str, str, int], Mapping[str, Any]],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, str]]]:
    datasets: dict[str, tuple[PrecomputedFeatureDataset, PrecomputedFeatureDataset]] = {}
    loaders: dict[str, tuple[DataLoader, DataLoader]] = {}
    feature_hashes: dict[str, dict[str, str]] = {}
    per_run: list[dict[str, Any]] = []
    per_episode: list[dict[str, Any]] = []
    use_amp = device.type == "cuda"

    for env in OCC_TO_CLEAN:
        sample = metadata[(env, "smtv3", 0)]
        paths = sample["feature_paths"]
        train = PrecomputedFeatureDataset(
            str(paths["train_feature_cache"]), str(paths["feature_manifest"])
        )
        validation = PrecomputedFeatureDataset(
            str(paths["val_feature_cache"]), str(paths["feature_manifest"])
        )
        if len(train) != 600 or len(validation) != 150:
            raise ValueError(f"{env}: feature episode counts differ from 600/150")
        if train.split != "train" or validation.split != "val":
            raise ValueError(f"{env}: feature split mismatch")
        datasets[env] = (train, validation)
        loaders[env] = (
            make_loader(train, batch_size, num_workers, device),
            make_loader(validation, batch_size, num_workers, device),
        )
        feature_hashes[env] = {
            name: sha256_file(path) for name, path in sorted(paths.items())
        }

    for env in OCC_TO_CLEAN:
        train_dataset, val_dataset = datasets[env]
        train_loader, val_loader = loaders[env]
        valid_mask = np.asarray(val_dataset.target_valid_mask, dtype=bool)
        for design in V3_DESIGNS:
            for seed in SEEDS:
                info = metadata[(env, design, seed)]
                checkpoint = torch.load(
                    info["model_path"], map_location="cpu", weights_only=False
                )
                model = build_model(info["cfg"], val_dataset.n_actions)
                model.load_state_dict(checkpoint["model_state_dict"], strict=True)
                model.to(device).eval()
                calibration_gates_full = collect_gates(model, train_loader, device)
                train_valid = np.asarray(train_dataset.target_valid_mask, dtype=bool)
                hidden = np.flatnonzero(~train_valid)
                if (not len(hidden) or
                        not np.array_equal(hidden, np.arange(hidden[0], hidden[-1] + 1))):
                    raise ValueError(f"{env}: train mask does not contain one contiguous blackout")
                first_post_time = int(hidden[-1] + 1)
                # m_0 is initialized directly from z_0, so gate t=0 is unused.  Prediction at the
                # first-post target depends only on updates through t=end-1.  Calibrating on this
                # exact causal prefix preserves its mean update mass while removing conditioning.
                calibration_gates = calibration_gates_full[:, 1:first_post_time]
                calibration_mean = float(calibration_gates.mean())
                episodes, validation_gates = evaluate_first_post(
                    model,
                    val_loader,
                    device,
                    use_amp,
                    calibration_mean,
                )
                run_row = summarize_gate_run(
                    info["run"],
                    env,
                    design,
                    seed,
                    calibration_gates,
                    validation_gates,
                    valid_mask,
                    episodes,
                    finite(
                        info["metrics"]["clean_mse_first_post"],
                        f"{info['run']} checkpoint first-post MSE",
                    ),
                    finite(
                        info["metrics"]["clean_input_mse_first_post"],
                        f"{info['run']} clean-input first-post MSE",
                    ),
                    1,
                    first_post_time,
                )
                per_run.append(run_row)
                for row in episodes:
                    per_episode.append(
                        {
                            "run": info["run"],
                            "env": env,
                            "design": design,
                            "seed": seed,
                            "calibration_gate_mean": calibration_mean,
                            "decision_gate_start": 1,
                            "decision_gate_end_exclusive": first_post_time,
                            **row,
                        }
                    )
                print(
                    f"gate audit {info['run']}: first-post={run_row['first_post_mse']:.6f} "
                    f"mean={run_row['first_post_mse_mean_gate']:.6f} "
                    f"causal-gap={run_row['val_causal_gate_visible_minus_black']:.4f} "
                    f"causal-auc={run_row['val_causal_gate_visible_auroc']:.4f}",
                    flush=True,
                )
                del model, checkpoint
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    per_run.sort(key=lambda row: (row["env"], row["design"], row["seed"]))
    per_episode.sort(
        key=lambda row: (row["env"], row["design"], row["seed"], row["episode"])
    )
    expected_v3 = set(itertools.product(OCC_TO_CLEAN, V3_DESIGNS, SEEDS))
    seen_v3 = {(row["env"], row["design"], row["seed"]) for row in per_run}
    if seen_v3 != expected_v3 or len(per_run) != NUM_GATE_RUNS:
        raise ValueError(
            f"gate audit did not produce the exact {NUM_GATE_RUNS} v3-mode cells"
        )
    expected_episode_rows = NUM_GATE_RUNS * 150
    if len(per_episode) != expected_episode_rows:
        raise ValueError(
            f"expected {expected_episode_rows} per-episode rows, got {len(per_episode)}"
        )
    return per_run, per_episode, feature_hashes


def grouped_gate_rows(per_run: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    identifier_fields = {
        "run",
        "env",
        "design",
        "seed",
        "n_calibration_episodes",
        "n_validation_episodes",
        "calibration_gate_start",
        "calibration_gate_end_exclusive",
        "decision_gate_start",
        "decision_gate_end_exclusive",
    }
    metrics = [key for key in per_run[0] if key not in identifier_fields]
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in per_run:
        groups[(row["env"], row["design"])].append(row)
        groups[("__overall__", row["design"])].append(row)
    output = []
    for (env, design), rows in sorted(groups.items()):
        expected_n = 25 if env == "__overall__" else 5
        if len(rows) != expected_n:
            raise ValueError(f"gate group {(env, design)} has {len(rows)} rows")
        out: dict[str, Any] = {
            "env": env,
            "design": design,
            "n_runs": len(rows),
        }
        for metric in metrics:
            values = [finite(row[metric], f"{env}/{design} {metric}") for row in rows]
            out[f"{metric}_mean"] = mean(values)
            out[f"{metric}_std"] = popstd(values)
        output.append(out)
    return output


def contrast_rows(per_run: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (row["env"], row["design"], int(row["seed"])): row for row in per_run
    }
    if len(lookup) != len(per_run):
        raise ValueError("duplicate v3 per-run gate row")
    definitions = (
        ("dynamic-static", "smtv3", "smtv3_static"),
        ("dynamic-old", "smtv3", "smtv3_old"),
        ("oracle-static", "smtv3_oracle", "smtv3_static"),
    )
    output: list[dict[str, Any]] = []
    for contrast, test_design, reference_design in definitions:
        for group_env in (*OCC_TO_CLEAN, "__overall__"):
            envs = tuple(OCC_TO_CLEAN) if group_env == "__overall__" else (group_env,)
            pairs = []
            for env in envs:
                for seed in SEEDS:
                    test = lookup[(env, test_design, seed)]
                    reference = lookup[(env, reference_design, seed)]
                    if abs(
                        float(test["last_visible_hold_mse"])
                        - float(reference["last_visible_hold_mse"])
                    ) > 2e-6:
                        raise ValueError(f"{env} seed {seed}: hold baseline differs by design")
                    pairs.append((test, reference))
            test_mse = [float(test["first_post_mse"]) for test, _ in pairs]
            reference_mse = [float(reference["first_post_mse"]) for _, reference in pairs]
            override_mse = [
                float(test["first_post_mse_mean_gate"]) for test, _ in pairs
            ]
            hold_mse = [float(test["last_visible_hold_mse"]) for test, _ in pairs]
            relative = [
                (reference_value - test_value) / reference_value
                for test_value, reference_value in zip(test_mse, reference_mse)
            ]
            intervention_relative = [
                (override_value - test_value) / test_value
                for override_value, test_value in zip(override_mse, test_mse)
            ]
            intervention_reference_normalized = [
                (override_value - test_value) / reference_value
                for override_value, test_value, reference_value in
                zip(override_mse, test_mse, reference_mse)
            ]
            test_mean = mean(test_mse)
            reference_mean = mean(reference_mse)
            override_mean = mean(override_mse)
            hold_mean = mean(hold_mse)
            gain = reference_mean - test_mean
            relative_gain = mean(relative)
            erasure = (mean(intervention_reference_normalized) / relative_gain
                       if abs(relative_gain) > 1e-12 else 0.0)
            output.append(
                {
                    "contrast": contrast,
                    "test_design": test_design,
                    "reference_design": reference_design,
                    "env": group_env,
                    "n_pairs": len(pairs),
                    "test_first_post_mse_mean": test_mean,
                    "reference_first_post_mse_mean": reference_mean,
                    "delta_first_post_mse_mean": test_mean - reference_mean,
                    "relative_improvement_mean": mean(relative),
                    "relative_improvement_of_means": gain / reference_mean,
                    "paired_wins": sum(a < b for a, b in zip(test_mse, reference_mse)),
                    "paired_ties": sum(a == b for a, b in zip(test_mse, reference_mse)),
                    "last_visible_hold_mse_mean": hold_mean,
                    "test_minus_hold_mse": test_mean - hold_mean,
                    "relative_improvement_vs_hold": (hold_mean - test_mean) / hold_mean,
                    "paired_wins_vs_hold": sum(a < b for a, b in zip(test_mse, hold_mse)),
                    "mean_gate_override_mse_mean": override_mean,
                    "mean_intervention_delta_mse": override_mean - test_mean,
                    "mean_intervention_relative_change": mean(intervention_relative),
                    "mean_intervention_erasure_fraction": erasure,
                }
            )
    return output


def invoke_mask_analysis(
    root: Path, device: torch.device, batch_size: int, num_workers: int
) -> list[dict[str, str]]:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "evaluate_shared_mask_generalization.py"),
        "--root",
        str(root),
        "--device",
        str(device),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--designs",
        *DESIGNS,
        "--seeds",
        *map(str, SEEDS),
    ]
    run_dependency(command)
    rows = read_csv_strict(root / "mask_generalization_per_run.csv")
    expected = set(itertools.product(OCC_TO_CLEAN, DESIGNS, SEEDS, (
        "original_10_16",
        *SHIFTED_CONDITIONS,
    )))
    found = {
        (row["env"], row["design"], int(row["seed"]), row["condition"])
        for row in rows
    }
    expected_rows = len(expected)
    if len(rows) != expected_rows or found != expected:
        raise ValueError(
            f"mask evaluator did not attest exact {NUM_RUNS}x4 cells: "
            f"{len(rows)}/{expected_rows}"
        )
    return rows


def build_decision(
    per_run: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
    convergence: Sequence[Mapping[str, Any]],
    mask_rows: Sequence[Mapping[str, str]],
    epochs: int,
) -> dict[str, Any]:
    gate_lookup = {
        (row["env"], row["design"], int(row["seed"])): row for row in per_run
    }
    contrast_lookup = {
        (row["contrast"], row["env"]): row for row in contrasts
    }
    overall = contrast_lookup[("dynamic-static", "__overall__")]
    dynamic_env_rel = {
        env: float(contrast_lookup[("dynamic-static", env)]["relative_improvement_of_means"])
        for env in OCC_TO_CLEAN
    }
    dynamic_hold_env = {
        env: (
            float(contrast_lookup[("dynamic-static", env)]["test_first_post_mse_mean"])
            < float(contrast_lookup[("dynamic-static", env)]["last_visible_hold_mse_mean"])
        )
        for env in OCC_TO_CLEAN
    }
    env_rel10_count = sum(value >= 0.10 for value in dynamic_env_rel.values())
    dynamic_hold_env_wins = sum(dynamic_hold_env.values())

    dynamic_rows = [row for row in per_run if row["design"] == "smtv3"]
    dynamic_auc = mean(
        [float(row["val_causal_gate_visible_auroc"]) for row in dynamic_rows]
    )
    dynamic_gap = mean(
        [float(row["val_causal_gate_visible_minus_black"]) for row in dynamic_rows]
    )

    # The parameter-matched old-erasing recurrence is an important descriptive control.  It was
    # not part of the locked decision bar, so expose it without changing any criterion or branch.
    dynamic_old_overall = contrast_lookup[("dynamic-old", "__overall__")]
    dynamic_old_env_rel = {
        env: float(
            contrast_lookup[("dynamic-old", env)]["relative_improvement_of_means"]
        )
        for env in OCC_TO_CLEAN
    }

    # PCA feature scales differ by environment; aggregate only paired relative effects, never raw
    # MSE across environments.
    clean_input_worsening = mean([
        (
            float(gate_lookup[(env, "smtv3", seed)]["clean_input_first_post_mse"])
            - float(gate_lookup[(env, "smtv3_static", seed)]["clean_input_first_post_mse"])
        ) / float(gate_lookup[(env, "smtv3_static", seed)]["clean_input_first_post_mse"])
        for env in OCC_TO_CLEAN for seed in SEEDS
    ])

    mask_lookup = {
        (row["env"], row["design"], int(row["seed"]), row["condition"]): finite(
            float(row["first_post_mse"]), "shifted-mask first-post MSE"
        )
        for row in mask_rows
    }
    shifted_relative: dict[str, float] = {}
    for condition in SHIFTED_CONDITIONS:
        dynamic = [
            mask_lookup[(env, "smtv3", seed, condition)]
            for env in OCC_TO_CLEAN
            for seed in SEEDS
        ]
        static = [
            mask_lookup[(env, "smtv3_static", seed, condition)]
            for env in OCC_TO_CLEAN
            for seed in SEEDS
        ]
        shifted_relative[condition] = mean([
            (static_value - dynamic_value) / static_value
            for dynamic_value, static_value in zip(dynamic, static)
        ])
    shifted_survives = all(value > 0.0 for value in shifted_relative.values())

    convergence_improvements = [
        float(row["relative_val_improvement_final_window"]) for row in convergence
    ]
    convergence_median = float(np.median(np.asarray(convergence_improvements)))
    convergence_max = max(convergence_improvements)
    converged = (
        convergence_median < CONVERGENCE_MEDIAN_MAX
        and convergence_max <= CONVERGENCE_CELL_MAX
    )

    # These adjacent-window and slope summaries are descriptive only.  In particular, neither
    # this block nor any of its outputs feeds ``converged`` or the GO/EXTEND/NO_GO branches.
    descriptive_mean_improvements = np.asarray(
        [
            float(row["descriptive_relative_val_mean_improvement"])
            for row in convergence
        ],
        dtype=np.float64,
    )
    descriptive_normalized_slopes = np.asarray(
        [
            float(row["descriptive_recent_val_slope_normalized_by_mean"])
            for row in convergence
        ],
        dtype=np.float64,
    )

    def descriptive_distribution(values: np.ndarray) -> dict[str, float]:
        if values.shape != (NUM_RUNS,) or not np.isfinite(values).all():
            raise ValueError("invalid descriptive convergence distribution")
        return {
            "p50": float(np.quantile(values, 0.50)),
            "p90": float(np.quantile(values, 0.90)),
            "p95": float(np.quantile(values, 0.95)),
            "max": float(values.max()),
        }

    descriptive_convergence = {
        "label": "descriptive_only_not_used_by_locked_decision",
        "used_by_locked_decision": False,
        "objective": "validation pred_loss",
        "num_cells": len(convergence),
        "previous_window_start_epoch": epochs - 2 * CONVERGENCE_WINDOW + 1,
        "previous_window_end_epoch": epochs - CONVERGENCE_WINDOW,
        "recent_window_start_epoch": epochs - CONVERGENCE_WINDOW + 1,
        "recent_window_end_epoch": epochs,
        "window_length_epochs": CONVERGENCE_WINDOW,
        "relative_window_mean_improvement": {
            "semantics": "(previous_window_mean - recent_window_mean) / previous_window_mean; positive means improvement",
            **descriptive_distribution(descriptive_mean_improvements),
        },
        "recent_linear_slope_normalized_by_mean": {
            "semantics": "OLS slope per epoch over the recent window divided by its mean; negative means improvement",
            **descriptive_distribution(descriptive_normalized_slopes),
        },
        "quantile_method": "NumPy linear interpolation",
    }

    oracle_env_rows = [contrast_lookup[("oracle-static", env)] for env in OCC_TO_CLEAN]
    oracle_static_env_wins = sum(
        float(row["test_first_post_mse_mean"]) < float(row["reference_first_post_mse_mean"])
        for row in oracle_env_rows)
    oracle_hold_env_wins = sum(
        float(row["test_first_post_mse_mean"]) < float(row["last_visible_hold_mse_mean"])
        for row in oracle_env_rows)
    oracle_beats_static = oracle_static_env_wins >= ORACLE_MIN_ENV_WINS
    oracle_beats_hold = oracle_hold_env_wins >= ORACLE_MIN_ENV_WINS

    overall_gain = float(overall["relative_improvement_mean"])
    mean_change = float(overall["mean_intervention_relative_change"])
    erasure = float(overall["mean_intervention_erasure_fraction"])
    paired_wins = int(overall["paired_wins"])
    go_criteria = {
        "dynamic_relative_improvement_ge_10pct_in_ge_4_of_5_envs": (
            env_rel10_count >= GO_MIN_ENV_REL10
        ),
        "dynamic_beats_static_in_ge_20_of_25_paired_cells": (
            paired_wins >= GO_MIN_PAIRED_WINS
        ),
        "dynamic_beats_last_visible_hold_in_ge_4_of_5_env_means": (
            dynamic_hold_env_wins >= GO_MIN_HOLD_ENV_WINS
        ),
        "mean_gate_replacement_erases_ge_50pct_dynamic_gain": (
            erasure >= GO_MIN_ERASURE_FRACTION
        ),
        "dynamic_causal_gate_visible_auroc_ge_0_90": (
            dynamic_auc >= GO_MIN_GATE_AUROC
        ),
        "dynamic_causal_gate_visible_minus_black_gap_ge_0_20": (
            dynamic_gap >= GO_MIN_GATE_GAP
        ),
        "dynamic_gain_survives_all_shifted_masks": shifted_survives,
        "dynamic_clean_input_mse_worsening_le_5pct_vs_static": (
            clean_input_worsening <= GO_MAX_CLEAN_INPUT_WORSENING
        ),
        "final_10_epoch_convergence_gate_passes": converged,
    }
    no_go_triggers = {
        "dynamic_static_overall_gain_lt_5pct": (
            overall_gain < NO_GO_MAX_DYNAMIC_STATIC_GAIN
        ),
        "mean_gate_intervention_changes_primary_lt_3pct": (
            abs(mean_change) < NO_GO_MAX_MEAN_INTERVENTION_CHANGE
        ),
        "oracle_fails_to_beat_static_or_hold": not (
            oracle_beats_static and oracle_beats_hold
        ),
    }
    scientific_go = all(
        value for key, value in go_criteria.items()
        if key != "final_10_epoch_convergence_gate_passes"
    )
    if any(no_go_triggers.values()):
        decision = "NO_GO"
        action = "Do not advance SMT-v3 as the claimed selective-memory architecture."
    elif not converged:
        decision = "EXTEND"
        action = (
            f"Extend every cell beyond epoch {epochs}, then rerun the same screening analysis."
        )
    elif scientific_go:
        decision = "GO"
        action = "Advance SMT-v3 to control-return evaluation and submission-scale baselines."
    else:
        decision = "NO_GO"
        action = "The prospective internal GO bar was not met; retain the diagnostic result."

    return {
        "schema_version": 1,
        "decision": decision,
        "recommended_action": action,
        "factorial": {
            "environments": len(OCC_TO_CLEAN),
            "designs": list(DESIGNS),
            "seeds": list(SEEDS),
            "runs": NUM_RUNS,
            "epochs": epochs,
            "first_post_loss_weight": FIRST_POST_LOSS_WEIGHT,
        },
        "thresholds": {
            "go_min_envs_with_10pct_dynamic_static_gain": GO_MIN_ENV_REL10,
            "go_min_dynamic_static_paired_wins": GO_MIN_PAIRED_WINS,
            "go_min_envs_beating_hold": GO_MIN_HOLD_ENV_WINS,
            "go_min_mean_gate_erasure_fraction": GO_MIN_ERASURE_FRACTION,
            "go_min_causal_gate_auroc": GO_MIN_GATE_AUROC,
            "go_min_causal_gate_visible_minus_black_gap": GO_MIN_GATE_GAP,
            "go_max_clean_input_worsening": GO_MAX_CLEAN_INPUT_WORSENING,
            "oracle_min_env_wins_vs_static_and_hold": ORACLE_MIN_ENV_WINS,
            "no_go_dynamic_static_gain_below": NO_GO_MAX_DYNAMIC_STATIC_GAIN,
            "no_go_mean_intervention_change_below": NO_GO_MAX_MEAN_INTERVENTION_CHANGE,
            "convergence_median_improvement_below": CONVERGENCE_MEDIAN_MAX,
            "convergence_max_cell_improvement_at_most": CONVERGENCE_CELL_MAX,
        },
        "observed": {
            "dynamic_static_overall_relative_gain": overall_gain,
            "dynamic_static_env_relative_gains": dynamic_env_rel,
            "dynamic_static_envs_at_least_10pct": env_rel10_count,
            "dynamic_static_paired_wins": paired_wins,
            "dynamic_old_descriptive_only_not_used_by_locked_decision": {
                "overall_relative_gain": float(
                    dynamic_old_overall["relative_improvement_mean"]
                ),
                "env_relative_gains": dynamic_old_env_rel,
                "paired_wins": int(dynamic_old_overall["paired_wins"]),
                "n_pairs": int(dynamic_old_overall["n_pairs"]),
            },
            "dynamic_hold_env_wins": dynamic_hold_env_wins,
            "dynamic_hold_by_env": dynamic_hold_env,
            "mean_intervention_relative_change": mean_change,
            "mean_intervention_erasure_fraction": erasure,
            "dynamic_causal_gate_visible_auroc_mean": dynamic_auc,
            "dynamic_causal_gate_visible_minus_black_gap_mean": dynamic_gap,
            "shifted_mask_relative_gains": shifted_relative,
            "clean_input_relative_worsening_vs_static": clean_input_worsening,
            "oracle_beats_static_overall": oracle_beats_static,
            "oracle_beats_hold_overall": oracle_beats_hold,
            "oracle_static_env_wins": oracle_static_env_wins,
            "oracle_hold_env_wins": oracle_hold_env_wins,
            "final_10_epoch_median_relative_improvement": convergence_median,
            "final_10_epoch_max_relative_improvement": convergence_max,
        },
        "go_criteria": go_criteria,
        "no_go_triggers": no_go_triggers,
        "all_scientific_go_criteria_pass": scientific_go,
        "training_converged": converged,
        "descriptive_convergence_diagnostics": descriptive_convergence,
        "metric_semantics": {
            "decision_gate_interval": (
                "causal primary prefix t=1..first_post-1; t=0 is an unused warm start and "
                "t>=first_post cannot affect the primary prediction"
            ),
            "causal_gate_auroc_positive_class": "visible/update-allowed timestep",
            "full_sequence_gate_metrics": (
                "descriptive only; include causally irrelevant t=0 and t>=first_post and are "
                "never used by the locked decision"
            ),
            "calibration_gate_mean": (
                "all train episodes over causal primary prefix t=1..first_post-1; t=0 is an "
                "unused warm start and future/post-target gates are excluded"
            ),
            "causal_gate_temporal_std": "std across causal-prefix time of episode-mean gate",
            "causal_gate_input_std": (
                "mean across causal-prefix time of episode-to-episode gate std"
            ),
            "dynamic_old_contrast": (
                "descriptive parameter-matched recurrence control; not used by the locked "
                "decision criteria or branches"
            ),
            "mean_intervention_erasure_fraction": (
                "mean((dynamic_mean_gate_mse-dynamic_mse)/static_mse) / "
                "mean((static_mse-dynamic_mse)/static_mse), paired within environment/seed"
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    root = args.root
    device = resolve_device(args.device)
    print(f"SMT-v3 analysis device: {device}", flush=True)

    artifacts = validate_artifact_layout(root)
    invoke_common_analysis(root, args.epochs)
    metadata, convergence, checkpoint_hashes = validate_checkpoints(
        root, artifacts, args.epochs
    )
    per_run, per_episode, feature_hashes = evaluate_v3_gates(
        metadata, device, args.batch_size, args.num_workers
    )
    grouped = grouped_gate_rows(per_run)
    contrasts = contrast_rows(per_run)
    mask_rows = invoke_mask_analysis(
        root, device, args.batch_size, args.num_workers
    )
    decision = build_decision(per_run, contrasts, convergence, mask_rows, args.epochs)

    output_rows = {
        "v3_gate_per_run.csv": per_run,
        "v3_gate_grouped.csv": grouped,
        "v3_gate_per_episode.csv": per_episode,
        "v3_contrasts.csv": contrasts,
        "convergence.csv": convergence,
    }
    for name, rows in output_rows.items():
        atomic_write_csv(root / name, rows)
    atomic_write_json(root / "decision.json", decision)

    dependency_outputs = (
        "per_run.csv",
        "grouped.csv",
        "paired_deltas.csv",
        "paired_grouped.csv",
        "mask_generalization_per_run.csv",
        "mask_generalization_grouped.csv",
    )
    output_names = (*output_rows, "decision.json", *dependency_outputs)
    output_hashes = {}
    for name in output_names:
        path = root / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing analysis output before manifest: {path}")
        output_hashes[name] = sha256_file(path)
    manifest = {
        "schema_version": 1,
        "protocol": {
            "environments": list(OCC_TO_CLEAN),
            "designs": list(DESIGNS),
            "v3_gate_designs": list(V3_DESIGNS),
            "seeds": list(SEEDS),
            "epochs": args.epochs,
            "first_post_loss_weight": FIRST_POST_LOSS_WEIGHT,
            "num_checkpoints": NUM_RUNS,
            "num_gate_checkpoints": len(per_run),
            "num_per_episode_rows": len(per_episode),
            "analysis_device": str(device),
            "batch_size": args.batch_size,
        },
        "checkpoint_hashes": checkpoint_hashes,
        "feature_artifact_hashes": feature_hashes,
        "analysis_code_hashes": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path)
            for path in (
                Path(__file__).resolve(),
                REPO_ROOT / "scripts" / "analyze_shared_clean_occlusion.py",
                REPO_ROOT / "scripts" / "evaluate_shared_mask_generalization.py",
                REPO_ROOT / "lewm" / "data.py",
                REPO_ROOT / "lewm" / "models" / "memory.py",
                REPO_ROOT / "lewm" / "models" / "memory_model.py",
            )
        },
        # These hashes attest the producer sources present when this analysis was finalized.
        # Checkpoints do not yet embed them, so clear-start/uninterrupted-run provenance remains
        # a separately documented limitation rather than a cryptographic checkpoint binding.
        "producer_code_hashes": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path)
            for path in (
                REPO_ROOT / "scripts" / "train_popgym.py",
                REPO_ROOT / "scripts" / "run_smt_v3.sh",
            )
        },
        "output_hashes": output_hashes,
    }
    atomic_write_json(root / "v3_manifest.json", manifest)
    print(
        f"SMT-v3 decision: {decision['decision']} -- {decision['recommended_action']}",
        flush=True,
    )
    print(
        f"validated {NUM_RUNS} checkpoints; wrote 5 atomic CSVs, decision.json, "
        "and v3_manifest.json",
        flush=True,
    )


if __name__ == "__main__":
    main()
