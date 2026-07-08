#!/usr/bin/env python3
"""Train one frozen-host carrier cell for a locked semantic PushT task."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.frozen_swap_carriers import (  # noqa: E402
    FROZEN_CARRIER_NAMES,
    make_frozen_carrier,
    parameter_report,
)
from lewm.models.official_lewm_pusht import (  # noqa: E402
    load_official_pusht_checkpoint,
)
from lewm.official_tasks.artifacts import (  # noqa: E402
    atomic_text,
    sha256_file,
    stable_json,
)
from lewm.official_tasks.pusht_pipeline import (  # noqa: E402
    aligned_pusht_latents,
    load_pusht_base_cache,
    load_pusht_task_cache,
    pusht_admission_path,
    pusht_carrier_directory,
    pusht_task_manifest_path,
    pusht_task_spec,
)
from lewm.official_tasks.pusht_spec import (  # noqa: E402
    DEFAULT_PUSHT_LOCK,
    DEFAULT_PUSHT_SPEC,
    load_locked_pusht_spec,
    pusht_lock_receipt,
    resolve_pusht_path,
    validate_pusht_device,
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--arm", required=True, choices=FROZEN_CARRIER_NAMES)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", required=True)
    parser.add_argument("--spec", type=Path, default=DEFAULT_PUSHT_SPEC)
    parser.add_argument("--lock", type=Path, default=DEFAULT_PUSHT_LOCK)
    parser.add_argument(
        "--execute", action="store_true",
        help="required acknowledgement before training or writing artifacts")
    return parser.parse_args(argv)


def _state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _configure(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def _load_admitted(spec: dict, task_key: str, split: str) -> dict:
    task_manifest_path = pusht_task_manifest_path(spec, task_key)
    admission_path = pusht_admission_path(spec, task_key)
    if not task_manifest_path.is_file() or not admission_path.is_file():
        raise FileNotFoundError(f"{task_key} has no completed frozen admission")
    manifest = json.loads(task_manifest_path.read_text())
    admission = json.loads(admission_path.read_text())
    if manifest.get("formal_lock") != pusht_lock_receipt(spec) \
            or admission.get("formal_lock") != pusht_lock_receipt(spec):
        raise ValueError("task cache/admission uses a different formal lock")
    receipt = manifest.get("admission", {})
    if receipt.get("sha256") != sha256_file(admission_path) \
            or receipt.get("admitted") is not True \
            or admission.get("admitted") is not True:
        raise RuntimeError(f"{task_key} did not pass every frozen admission")
    base, base_meta = load_pusht_base_cache(spec, split)
    task, task_meta = load_pusht_task_cache(spec, task_key, split)
    z = aligned_pusht_latents(
        base, task, spec["sequence"]["cue_start"],
        spec["sequence"]["cue_length"])
    return {
        **base,
        **task,
        "z": z,
        "base_meta": base_meta,
        "task_meta": task_meta,
    }


def _train_epoch(model: torch.nn.Module, carrier: torch.nn.Module,
                 optimizer: torch.optim.Optimizer, z: np.ndarray,
                 actions: np.ndarray, batch_size: int, windows: int,
                 rng: np.random.Generator, device: torch.device) -> float:
    carrier.train()
    history = 3
    valid_starts = z.shape[1] - history
    order = rng.permutation(len(z))
    losses = []
    for offset in range(0, len(order), batch_size):
        rows = order[offset:offset + batch_size]
        if len(rows) < 4:
            continue
        z_batch = torch.from_numpy(z[rows]).to(device)
        action_batch = torch.from_numpy(actions[rows]).to(device)
        starts = rng.choice(
            valid_starts, min(windows, valid_starts), replace=False)
        optimizer.zero_grad(set_to_none=True)
        fused = carrier(z_batch, action_batch).z_tilde
        latent = torch.cat([
            fused[:, int(start):int(start) + history]
            for start in starts], dim=0)
        action = torch.cat([
            action_batch[:, int(start):int(start) + history]
            for start in starts], dim=0)
        target = torch.cat([
            z_batch[:, int(start) + history]
            for start in starts], dim=0)
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
def _carrier_prior(carrier: torch.nn.Module, z: np.ndarray,
                   actions: np.ndarray, device: torch.device,
                   batch_size: int = 64) -> np.ndarray:
    carrier.eval()
    values = []
    for start in range(0, len(z), batch_size):
        output = carrier(
            torch.from_numpy(z[start:start + batch_size]).to(device),
            torch.from_numpy(actions[start:start + batch_size]).to(device))
        values.append(output.prior_read.float().cpu().numpy())
    return np.concatenate(values)


def _final_probe(train: dict, train_prior: np.ndarray,
                 validation: dict, validation_prior: np.ndarray,
                 spec: dict) -> dict:
    context = spec["sequence"]["final_context_indices"]
    decision = spec["sequence"]["decision_index"]
    if max(context) >= decision or decision != 19:
        raise ValueError("decision observation entered the formal endpoint")

    def features(data: dict, prior: np.ndarray) -> np.ndarray:
        raw = data["z"][:, context].reshape(len(prior), -1)
        return np.concatenate((raw, prior[:, decision]), axis=1)

    train_x = features(train, train_prior)
    validation_x = features(validation, validation_prior)
    train_y = train["labels"]
    validation_y = validation["labels"]
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=4000, random_state=0),
    )
    classifier.fit(train_x, train_y)
    prediction = classifier.predict(validation_x)
    classes = len(np.unique(train_y))
    return {
        "metric": "balanced_accuracy",
        "balanced_accuracy": float(balanced_accuracy_score(
            validation_y, prediction)),
        "accuracy": float(np.mean(validation_y == prediction)),
        "chance": 1.0 / classes,
        "fit_episodes": int(len(train_y)),
        "validation_episodes": int(len(validation_y)),
        "endpoint": {
            "decision_index": decision,
            "decision_observation_excluded": True,
            "raw_context_indices": context,
            "carrier_prior_index": decision,
            "prior_conditioning": (
                "z[0:19] and action[0:19], before consuming z[19]"),
            "feature": "concat(z[:,16:19], prior_read[:,19])",
        },
    }


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    spec = load_locked_pusht_spec(args.spec, args.lock)
    validate_pusht_device(args.device)
    task = pusht_task_spec(spec, args.task)
    training = spec["carrier_training"]
    if args.arm not in training["arms"] or args.seed not in training["seeds"]:
        raise ValueError("requested carrier cell lies outside the locked grid")
    if not args.execute:
        raise RuntimeError("formal PushT carrier training requires --execute")
    train = _load_admitted(spec, args.task, "train")
    validation = _load_admitted(spec, args.task, "validation")
    output = pusht_carrier_directory(spec, args.task, args.arm, args.seed)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite carrier cell {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("formal PushT carrier training requires CUDA")

    _configure(args.seed)
    device = torch.device(args.device)
    bundle = resolve_pusht_path(spec["official_host"]["bundle_path"])
    model = load_official_pusht_checkpoint(bundle, device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    host_before = _state_digest(model)
    carrier = make_frozen_carrier(args.arm, 192, 10).to(device)
    epochs = int(training["epochs"])
    rows = []
    if carrier.parameter_count():
        optimizer = torch.optim.AdamW(
            carrier.parameters(), lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs)
        rng = np.random.default_rng(471_000 + args.seed)
        for epoch in range(1, epochs + 1):
            loss = _train_epoch(
                model, carrier, optimizer, train["z"], train["actions"],
                int(training["batch_size"]),
                int(training["windows_per_episode"]), rng, device)
            scheduler.step()
            rows.append({"epoch": epoch, "loss": loss,
                         "lr": optimizer.param_groups[0]["lr"]})
    train_prior = _carrier_prior(
        carrier, train["z"], train["actions"], device)
    validation_prior = _carrier_prior(
        carrier, validation["z"], validation["actions"], device)
    probe = _final_probe(train, train_prior, validation,
                         validation_prior, spec)
    host_after = _state_digest(model)
    if host_before != host_after:
        raise RuntimeError("frozen official PushT host changed during training")

    output.mkdir(parents=True, exist_ok=False)
    history_path = output / "history.csv"
    with history_path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("epoch", "loss", "lr"))
        writer.writeheader()
        writer.writerows(rows)
    metrics = {
        "schema": "official_pusht_carrier_metrics_v1",
        "task_key": args.task,
        "semantic_name": task["display_name"],
        "arm": args.arm,
        "seed": args.seed,
        "formal_lock": pusht_lock_receipt(spec),
        "frozen_host_sha256_before": host_before,
        "frozen_host_sha256_after": host_after,
        "frozen_host_unchanged": True,
        "carrier_parameters": carrier.parameter_count(),
        "parameter_matching": parameter_report(192, 10),
        "carrier_config": carrier.describe(),
        "epochs": epochs if carrier.parameter_count() else 0,
        "final_train_loss": rows[-1]["loss"] if rows else None,
        "primary_probe": probe,
    }
    metrics_path = output / "metrics.json"
    metrics_hash = atomic_text(metrics_path, stable_json(metrics))
    checkpoint_path = output / "carrier.pt"
    torch.save({"carrier_state_dict": carrier.state_dict(),
                "metrics": metrics}, checkpoint_path)
    manifest = {
        "schema": "official_pusht_carrier_manifest_v1",
        "task_key": args.task,
        "semantic_name": task["display_name"],
        "arm": args.arm,
        "seed": args.seed,
        "formal_lock": pusht_lock_receipt(spec),
        "artifacts": {
            "metrics": {"path": metrics_path.name, "sha256": metrics_hash},
            "checkpoint": {"path": checkpoint_path.name,
                           "sha256": sha256_file(checkpoint_path)},
            "history": {"path": history_path.name,
                        "sha256": sha256_file(history_path)},
        },
    }
    atomic_text(output / "manifest.json", stable_json(manifest))


if __name__ == "__main__":
    main()
