#!/usr/bin/env python3
"""Train one existing frozen carrier on an admitted semantic capacity stage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.frozen_swap_carriers import (  # noqa: E402
    FROZEN_CARRIER_NAMES,
    make_frozen_carrier,
    parameter_report,
)
from lewm.models.official_lewm import (  # noqa: E402
    OFFICIAL_ACTION_DIM,
    OFFICIAL_EMBED_DIM,
    OFFICIAL_HISTORY,
    load_official_reacher_checkpoint,
)
from lewm.official_tasks.artifacts import (  # noqa: E402
    atomic_text,
    load_verified_npz,
    sha256_file,
    stable_json,
    write_npz,
)
from lewm.official_tasks.shell_game_pipeline import (  # noqa: E402
    admission_path,
    cache_manifest_path,
    cache_path,
    carrier_directory,
    lock_receipt,
    stage_contract,
)
from lewm.official_tasks.shell_game_spec import (  # noqa: E402
    DEFAULT_LOCK,
    DEFAULT_SPEC,
    load_locked_spec,
    resolve_path,
    validate_device,
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=("single-item", "two-item", "four-item"))
    parser.add_argument("--arm", required=True, choices=FROZEN_CARRIER_NAMES)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", required=True)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    return parser.parse_args(argv)


def _state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _configure_determinism(seed: int) -> None:
    """Fail-closed deterministic settings for direct or launched runs."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")


def _load_admitted_cache(spec: dict, stage: str, split: str) -> dict:
    manifest_path = cache_manifest_path(spec, stage)
    admission_file = admission_path(spec, stage)
    if not manifest_path.is_file() or not admission_file.is_file():
        raise FileNotFoundError(f"stage {stage} has no completed admission")
    manifest = json.loads(manifest_path.read_text())
    admission = json.loads(admission_file.read_text())
    receipt = manifest.get("admission", {})
    if receipt.get("sha256") != sha256_file(admission_file) \
            or receipt.get("admitted") is not True \
            or admission.get("admitted") is not True:
        raise RuntimeError(f"stage {stage} did not pass frozen admission")
    if manifest.get("formal_lock") != lock_receipt(spec) \
            or admission.get("formal_lock") != lock_receipt(spec):
        raise ValueError("cache/admission was produced under a different lock")
    arrays, sidecar = load_verified_npz(cache_path(spec, stage, split))
    required = {
        "z", "actions", "initial_slots", "final_slots", "cue_on",
        "cue_off", "swap_pairs", "shuffle_off",
    }
    if set(arrays) != required:
        raise ValueError(f"cache fields differ: {sorted(arrays)}")
    if sidecar.get("schema") != "official_shell_game_cache_v1" \
            or sidecar.get("stage") != stage \
            or sidecar.get("split") != split \
            or sidecar.get("formal_lock") != lock_receipt(spec):
        raise ValueError("cache metadata differs from formal contract")
    if arrays["z"].shape[-1] != OFFICIAL_EMBED_DIM \
            or arrays["actions"].shape[-1] != OFFICIAL_ACTION_DIM:
        raise ValueError("cache does not use official latent/action dimensions")
    arrays["meta"] = sidecar
    arrays["admission"] = admission
    return arrays


def _sampled_windows(z_tilde: torch.Tensor, actions: torch.Tensor,
                     targets: torch.Tensor, starts: np.ndarray
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    latent = torch.stack([
        z_tilde[:, int(start):int(start) + OFFICIAL_HISTORY]
        for start in starts
    ], dim=1)
    action = torch.stack([
        actions[:, int(start):int(start) + OFFICIAL_HISTORY]
        for start in starts
    ], dim=1)
    target = torch.stack([
        targets[:, int(start) + OFFICIAL_HISTORY]
        for start in starts
    ], dim=1)
    batch, windows = latent.shape[:2]
    return (
        latent.reshape(batch * windows, OFFICIAL_HISTORY, -1),
        action.reshape(batch * windows, OFFICIAL_HISTORY, -1),
        target.reshape(batch * windows, -1),
    )


def _train_epoch(model, carrier, optimizer, z: np.ndarray,
                 actions: np.ndarray, batch_size: int, windows: int,
                 rng: np.random.Generator, device: torch.device) -> float:
    carrier.train()
    order = rng.permutation(len(z))
    losses = []
    for offset in range(0, len(order), batch_size):
        index = order[offset:offset + batch_size]
        if len(index) < 4:
            continue
        z_batch = torch.from_numpy(z[index]).to(device)
        action_batch = torch.from_numpy(actions[index]).to(device)
        starts = rng.choice(
            z.shape[1] - OFFICIAL_HISTORY,
            size=min(windows, z.shape[1] - OFFICIAL_HISTORY),
            replace=False,
        )
        optimizer.zero_grad(set_to_none=True)
        output = carrier(z_batch, action_batch)
        latent, action, target = _sampled_windows(
            output.z_tilde, action_batch, z_batch, starts)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model.predict(latent, action)[:, -1]
            loss = F.mse_loss(prediction.float(), target.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(carrier.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    if not losses:
        raise RuntimeError("carrier epoch produced no optimization batches")
    return float(np.mean(losses))


@torch.no_grad()
def _carrier_outputs(carrier, z: np.ndarray, actions: np.ndarray,
                     device: torch.device, batch_size: int = 32
                     ) -> tuple[np.ndarray, np.ndarray]:
    carrier.eval()
    fused, prior = [], []
    for offset in range(0, len(z), batch_size):
        z_batch = torch.from_numpy(z[offset:offset + batch_size]).to(device)
        action_batch = torch.from_numpy(
            actions[offset:offset + batch_size]).to(device)
        output = carrier(z_batch, action_batch)
        fused.append(output.z_tilde.float().cpu().numpy())
        prior.append(output.prior_read.float().cpu().numpy())
    return np.concatenate(fused), np.concatenate(prior)


@torch.no_grad()
def _prediction_mse(model, fused: np.ndarray, actions: np.ndarray,
                    targets: np.ndarray, device: torch.device,
                    chunk: int = 1024) -> float:
    latent, action, target = [], [], []
    for start in range(targets.shape[1] - OFFICIAL_HISTORY):
        latent.append(fused[:, start:start + OFFICIAL_HISTORY])
        action.append(actions[:, start:start + OFFICIAL_HISTORY])
        target.append(targets[:, start + OFFICIAL_HISTORY])
    latent_array = np.stack(latent, axis=1).reshape(
        -1, OFFICIAL_HISTORY, OFFICIAL_EMBED_DIM)
    action_array = np.stack(action, axis=1).reshape(
        -1, OFFICIAL_HISTORY, OFFICIAL_ACTION_DIM)
    target_array = np.stack(target, axis=1).reshape(-1, OFFICIAL_EMBED_DIM)
    squared, count = 0.0, 0
    for offset in range(0, len(latent_array), chunk):
        x = torch.from_numpy(latent_array[offset:offset + chunk]).to(device)
        a = torch.from_numpy(action_array[offset:offset + chunk]).to(device)
        y = torch.from_numpy(target_array[offset:offset + chunk]).to(device)
        prediction = model.predict(x, a)[:, -1]
        squared += float((prediction.float() - y.float()).square().sum())
        count += y.numel()
    return squared / count


def _decision_features(data: dict, prior: np.ndarray) -> tuple[np.ndarray, dict]:
    z = np.asarray(data["z"], dtype=np.float64)
    prior = np.asarray(prior, dtype=np.float64)
    contract = stage_contract(data["meta"]["stage"])
    if prior.shape != z.shape:
        raise ValueError("prior and latent arrays must have identical shape")
    indices = list(contract.final_context_indices)
    if max(indices) >= contract.decision_index:
        raise AssertionError("decision observation entered primary features")
    context = z[:, indices].reshape(len(z), -1)
    features = np.concatenate(
        [context, prior[:, contract.decision_index]], axis=1)
    endpoint = {
        "schema": "official_shell_game_final_endpoint_v1",
        "decision_observation_index": contract.decision_index,
        "raw_context_indices": indices,
        "prior_index": contract.decision_index,
        "prior_timing": "before consuming the decision observation",
        "decision_observation_excluded": True,
        "temporal_aggregation": False,
        "feature": "concat(flatten(z[:,60:63]), prior_read[:,63])",
        "feature_dim": int(features.shape[1]),
    }
    return features, endpoint


def _primary_probe(train: dict, train_prior: np.ndarray,
                   validation: dict, validation_prior: np.ndarray) -> dict:
    train_x, endpoint = _decision_features(train, train_prior)
    validation_x, validation_endpoint = _decision_features(
        validation, validation_prior)
    if endpoint != validation_endpoint:
        raise ValueError("train and validation endpoint contracts differ")
    train_y = np.asarray(train["final_slots"], dtype=np.int64)
    validation_y = np.asarray(validation["final_slots"], dtype=np.int64)
    if train_y.ndim != 2 or validation_y.shape[1] != train_y.shape[1]:
        raise ValueError("final slot targets must be (E,capacity)")
    predictions = np.empty_like(validation_y)
    per_item = []
    for item in range(train_y.shape[1]):
        if set(np.unique(train_y[:, item])) != {0, 1, 2} \
                or set(np.unique(validation_y[:, item])) != {0, 1, 2}:
            raise ValueError("every item must contain all three target slots")
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0, solver="lbfgs", max_iter=4000, random_state=0),
        )
        classifier.fit(train_x, train_y[:, item])
        predictions[:, item] = classifier.predict(validation_x)
        per_item.append({
            "item_index": item,
            "balanced_accuracy": float(balanced_accuracy_score(
                validation_y[:, item], predictions[:, item])),
            "accuracy": float(accuracy_score(
                validation_y[:, item], predictions[:, item])),
            "validation_class_counts": np.bincount(
                validation_y[:, item], minlength=3).astype(int).tolist(),
        })
    capacity = train_y.shape[1]
    balanced = [item["balanced_accuracy"] for item in per_item]
    return {
        "metric": "mean_per_item_balanced_accuracy",
        "mean_per_item_balanced_accuracy": float(np.mean(balanced)),
        "minimum_per_item_balanced_accuracy": float(np.min(balanced)),
        "per_item": per_item,
        "exact_set_accuracy": float(np.mean(np.all(
            predictions == validation_y, axis=1))),
        "per_item_chance": 1.0 / 3.0,
        "exact_set_chance": float(3 ** -capacity),
        "fit_episodes": int(len(train_y)),
        "validation_episodes": int(len(validation_y)),
        "readout": (
            "one train-bank StandardScaler+LogisticRegression per item; "
            "fixed validation bank"),
        "endpoint_contract": endpoint,
    }


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    spec = load_locked_spec(args.spec, args.lock)
    validate_device(args.device)
    training = spec["carrier_training"]
    if args.arm not in training["arms"] or args.seed not in training["seeds"]:
        raise ValueError("requested carrier cell lies outside the locked grid")

    # Admission and cache integrity are checked before output directories or
    # GPU models are created.
    train = _load_admitted_cache(spec, args.stage, "train")
    validation = _load_admitted_cache(spec, args.stage, "validation")
    output = carrier_directory(spec, args.stage, args.arm, args.seed)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite carrier directory {output}")

    weights = resolve_path(spec["official_host"]["weights_path"])
    if sha256_file(weights) != spec["official_host"]["weights_sha256"]:
        raise ValueError("official checkpoint differs from locked hash")
    if not torch.cuda.is_available():
        raise RuntimeError("formal carrier training requires an allowed CUDA device")
    _configure_determinism(args.seed)
    device = torch.device(args.device)
    model = load_official_reacher_checkpoint(weights, device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    host_before = _state_digest(model)
    carrier = make_frozen_carrier(
        args.arm, OFFICIAL_EMBED_DIM, OFFICIAL_ACTION_DIM).to(device)

    epochs = int(training["epochs"])
    batch_size = int(training["batch_size"])
    rows = []
    output.mkdir(parents=True, exist_ok=False)
    history_path = output / "history.csv"
    with history_path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("epoch", "loss", "lr"))
        writer.writeheader()
        if carrier.parameter_count() > 0:
            optimizer = torch.optim.AdamW(
                carrier.parameters(), lr=float(training["learning_rate"]),
                weight_decay=float(training["weight_decay"]))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs)
            rng = np.random.default_rng(71_000 + args.seed)
            for epoch in range(1, epochs + 1):
                loss = _train_epoch(
                    model, carrier, optimizer,
                    np.asarray(train["z"], dtype=np.float32),
                    np.asarray(train["actions"], dtype=np.float32),
                    batch_size, int(training["windows_per_episode"]),
                    rng, device)
                scheduler.step()
                row = {"epoch": epoch, "loss": loss,
                       "lr": optimizer.param_groups[0]["lr"]}
                rows.append(row)
                writer.writerow(row)
                stream.flush()
                if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
                    print(f"[shell-game-carrier] {args.stage}/{args.arm}/"
                          f"seed-{args.seed} {epoch}/{epochs} loss={loss:.6f}")

    _, train_prior = _carrier_outputs(
        carrier, np.asarray(train["z"], dtype=np.float32),
        np.asarray(train["actions"], dtype=np.float32), device)
    fused, validation_prior = _carrier_outputs(
        carrier, np.asarray(validation["z"], dtype=np.float32),
        np.asarray(validation["actions"], dtype=np.float32), device)
    prediction_mse = _prediction_mse(
        model, fused, np.asarray(validation["actions"], dtype=np.float32),
        np.asarray(validation["z"], dtype=np.float32), device)
    probe = _primary_probe(train, train_prior, validation, validation_prior)
    host_after = _state_digest(model)
    if host_before != host_after:
        raise RuntimeError("frozen official host changed during carrier training")

    metrics = {
        "schema": "official_shell_game_carrier_metrics_v1",
        "study": spec["study"],
        "stage": args.stage,
        "display_name": stage_contract(args.stage).stage.display_name,
        "capacity": stage_contract(args.stage).stage.capacity,
        "arm": args.arm,
        "seed": args.seed,
        "formal_lock": lock_receipt(spec),
        "official_checkpoint_sha256": spec["official_host"]["weights_sha256"],
        "official_host_state_sha256_before": host_before,
        "official_host_state_sha256_after": host_after,
        "frozen_host_unchanged": True,
        "carrier_parameters": carrier.parameter_count(),
        "parameter_matching": parameter_report(
            OFFICIAL_EMBED_DIM, OFFICIAL_ACTION_DIM),
        "carrier_config": carrier.describe(),
        "epochs": epochs if carrier.parameter_count() else 0,
        "batch_size": batch_size,
        "learning_rate": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "final_train_loss": rows[-1]["loss"] if rows else None,
        "validation_next_latent_mse": prediction_mse,
        "primary_probe": probe,
        "admission_sha256": sha256_file(admission_path(spec, args.stage)),
        "source_caches": {
            split: {
                "path": str(cache_path(spec, args.stage, split)),
                "sha256": sha256_file(cache_path(spec, args.stage, split)),
            }
            for split in ("train", "validation")
        },
    }
    metrics_path = output / "metrics.json"
    metrics_hash = atomic_text(metrics_path, stable_json(metrics))
    checkpoint_path = output / "carrier.pt"
    torch.save({
        "carrier_state_dict": carrier.state_dict(),
        "metrics": metrics,
    }, checkpoint_path)
    export_path = output / "validation_export.npz"
    export_hash = write_npz(export_path, {
        "prior_read": validation_prior,
        "z": np.asarray(validation["z"], dtype=np.float32),
        "actions": np.asarray(validation["actions"], dtype=np.float32),
        "final_slots": np.asarray(validation["final_slots"], dtype=np.int64),
        "meta_json": np.asarray(json.dumps({
            "stage": args.stage,
            "display_name": stage_contract(args.stage).stage.display_name,
            "arm": args.arm,
            "seed": args.seed,
        }, sort_keys=True)),
    })
    manifest_path = output / "manifest.json"
    atomic_text(manifest_path, stable_json({
        "schema": "official_shell_game_carrier_manifest_v1",
        "study": spec["study"],
        "stage": args.stage,
        "arm": args.arm,
        "seed": args.seed,
        "formal_lock": lock_receipt(spec),
        "artifacts": {
            "metrics": {
                "path": metrics_path.name,
                "sha256": metrics_hash,
            },
            "checkpoint": {
                "path": checkpoint_path.name,
                "sha256": sha256_file(checkpoint_path),
            },
            "validation_export": {
                "path": export_path.name,
                "sha256": export_hash,
            },
            "history": {
                "path": history_path.name,
                "sha256": sha256_file(history_path),
            },
        },
    }))
    print(json.dumps({
        "manifest": str(manifest_path),
        "metrics": str(metrics_path),
        "mean_per_item_balanced_accuracy":
            probe["mean_per_item_balanced_accuracy"],
        "exact_set_accuracy": probe["exact_set_accuracy"],
        "per_item_chance": probe["per_item_chance"],
        "exact_set_chance": probe["exact_set_chance"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
