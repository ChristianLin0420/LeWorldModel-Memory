#!/usr/bin/env python3
"""Train publication carriers against one frozen official LeWM host.

The released SIGReg Reacher encoder and predictor remain byte-identical.
Only a small causal carrier is optimized through the frozen next-latent loss.
All arms receive the same final H=3 raw-latent context at evaluation; they
differ only in the pre-observation persistent-state read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.frozen_swap_carriers import (
    FROZEN_CARRIER_NAMES,
    make_frozen_carrier,
    parameter_report,
)
from lewm.models.official_lewm import (OFFICIAL_ACTION_DIM,
                                        OFFICIAL_EMBED_DIM,
                                        OFFICIAL_HISTORY,
                                        load_official_reacher_checkpoint)
from scripts.paper_a_robustness_spec import (load_locked_spec,
                                             resolve_spec_path,
                                             validate_device)

TASKS = ("t1", "t3", "t4")
WINDOWS_PER_EPISODE = 8
RIDGE_ALPHAS = np.logspace(-3, 3, 7)
CATEGORICAL_PROBE_SCHEMA = "official-frozen-decision-endpoint-v2"
TRAJECTORY_PROBE_SCHEMA = "official-frozen-trajectory-diagnostic-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cache(path: Path) -> dict:
    with np.load(path) as source:
        data = {key: source[key] for key in source.files}
    data["meta"] = json.loads(str(data.pop("meta_json", np.array("{}"))))
    required = {"z", "actions", "xi"}
    missing = required.difference(data)
    if missing:
        raise ValueError(f"{path} missing keys {sorted(missing)}")
    if data["z"].shape[-1] != OFFICIAL_EMBED_DIM:
        raise ValueError("cache is not official 192-D LeWM latent data")
    if data["actions"].shape[-1] != OFFICIAL_ACTION_DIM:
        raise ValueError("cache does not use the official 10-D action block")
    return data


def state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def sampled_windows(z_tilde: torch.Tensor, actions: torch.Tensor,
                    targets: torch.Tensor, starts: np.ndarray
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    latent = torch.stack(
        [z_tilde[:, int(start):int(start) + OFFICIAL_HISTORY]
         for start in starts], dim=1)
    action = torch.stack(
        [actions[:, int(start):int(start) + OFFICIAL_HISTORY]
         for start in starts], dim=1)
    target = torch.stack(
        [targets[:, int(start) + OFFICIAL_HISTORY]
         for start in starts], dim=1)
    batch, windows = latent.shape[:2]
    return (latent.reshape(batch * windows, OFFICIAL_HISTORY, -1),
            action.reshape(batch * windows, OFFICIAL_HISTORY, -1),
            target.reshape(batch * windows, -1))


def train_epoch(model, carrier, optimizer, z: np.ndarray,
                actions: np.ndarray, batch_size: int,
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
            size=min(WINDOWS_PER_EPISODE,
                     z.shape[1] - OFFICIAL_HISTORY),
            replace=False)
        optimizer.zero_grad(set_to_none=True)
        output = carrier(z_batch, action_batch)
        latent, action, target = sampled_windows(
            output.z_tilde, action_batch, z_batch, starts)
        amp = (torch.autocast("cuda", dtype=torch.bfloat16)
               if device.type == "cuda"
               else torch.autocast("cpu", enabled=False))
        with amp:
            prediction = model.predict(latent, action)[:, -1]
            loss = F.mse_loss(prediction.float(), target.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(carrier.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


@torch.no_grad()
def carrier_outputs(carrier, z: np.ndarray, actions: np.ndarray,
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
def prediction_mse(model, fused: np.ndarray, actions: np.ndarray,
                   targets: np.ndarray, device: torch.device,
                   chunk: int = 1024) -> float:
    latent_parts, action_parts, target_parts = [], [], []
    for start in range(targets.shape[1] - OFFICIAL_HISTORY):
        latent_parts.append(fused[:, start:start + OFFICIAL_HISTORY])
        action_parts.append(actions[:, start:start + OFFICIAL_HISTORY])
        target_parts.append(targets[:, start + OFFICIAL_HISTORY])
    latent = np.stack(latent_parts, axis=1).reshape(
        -1, OFFICIAL_HISTORY, OFFICIAL_EMBED_DIM)
    action = np.stack(action_parts, axis=1).reshape(
        -1, OFFICIAL_HISTORY, OFFICIAL_ACTION_DIM)
    target = np.stack(target_parts, axis=1).reshape(-1, OFFICIAL_EMBED_DIM)
    squared, count = 0.0, 0
    for offset in range(0, len(latent), chunk):
        x = torch.from_numpy(latent[offset:offset + chunk]).to(device)
        a = torch.from_numpy(action[offset:offset + chunk]).to(device)
        y = torch.from_numpy(target[offset:offset + chunk]).to(device)
        prediction = model.predict(x, a)[:, -1]
        squared += float((prediction.float() - y.float()).square().sum())
        count += y.numel()
    return squared / count


def _categorical_endpoint(data: dict) -> tuple[int, list[int]]:
    """Return the registered decision index and its legal raw context.

    The categorical publication banks place the decision observation at the
    final frame.  A decision-time read may consume observations strictly
    before that frame and ``prior_read[:, q]``, which is computed before
    consuming ``z[:, q]``.  It may not consume the decision observation
    itself.
    """

    z = np.asarray(data["z"], dtype=np.float64)
    if z.ndim != 3 or z.shape[1] <= OFFICIAL_HISTORY:
        raise ValueError(
            "categorical probe requires z shaped (E,L,D) with L > history")
    decision_index = int(z.shape[1] - 1)
    indices = list(range(decision_index - OFFICIAL_HISTORY, decision_index))
    return decision_index, indices


def categorical_probe_contract(data: dict) -> dict:
    decision_index, indices = _categorical_endpoint(data)
    return {
        "schema": CATEGORICAL_PROBE_SCHEMA,
        "decision_observation_index": decision_index,
        "raw_context_history": OFFICIAL_HISTORY,
        "raw_context_indices": indices,
        "raw_context_slice": "z[:, q-H:q]",
        "raw_context_cutoff_exclusive": decision_index,
        "final_prior_index": decision_index,
        "final_prior_timing": "prior_read[:, q] before consuming z[:, q]",
        "feature_order": ["raw_predecision_context_flat", "final_preobservation_prior"],
        "current_observation_excluded": True,
        "future_observation_consumed": False,
        "temporal_aggregation": False,
    }


def categorical_features(data: dict, prior: np.ndarray) -> np.ndarray:
    """Registered final decision-time features with no temporal aggregation."""

    z = np.asarray(data["z"], dtype=np.float64)
    prior = np.asarray(prior, dtype=np.float64)
    decision_index, indices = _categorical_endpoint(data)
    if prior.shape != z.shape:
        raise ValueError(
            f"prior must match z shape {z.shape}, got {prior.shape}")
    context = z[:, indices].reshape(len(z), -1)
    return np.concatenate([context, prior[:, decision_index]], axis=1)


def categorical_trajectory_features(data: dict, prior: np.ndarray) -> np.ndarray:
    """Exploratory trajectory summary, never the publication primary endpoint."""

    z = np.asarray(data["z"], dtype=np.float64)
    prior = np.asarray(prior, dtype=np.float64)
    decision_index, indices = _categorical_endpoint(data)
    if prior.shape != z.shape:
        raise ValueError(
            f"prior must match z shape {z.shape}, got {prior.shape}")
    context = z[:, indices].reshape(len(z), -1)
    cue_off = np.asarray(data["event_cue_off"], dtype=np.int64)
    steps = np.arange(z.shape[1])[None]
    mask = ((steps >= (cue_off[:, None] + 2))
            & (steps <= decision_index))
    if np.any(mask.sum(axis=1) == 0):
        raise ValueError("trajectory probe has an empty post-cue prior interval")
    weights = mask / mask.sum(axis=1, keepdims=True)
    prior_mean = np.einsum("el,eld->ed", weights, prior)
    return np.concatenate(
        [context, prior_mean, prior[:, decision_index]], axis=1)


def _fit_categorical_probe(train_x: np.ndarray, train_y: np.ndarray,
                           val_x: np.ndarray, val_y: np.ndarray) -> dict:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=4000, C=1.0, solver="lbfgs",
                           random_state=0),
    )
    model.fit(train_x, train_y)
    score = float(accuracy_score(val_y, model.predict(val_x)))
    classes = int(np.max(train_y)) + 1
    return {"metric": "accuracy", "mean": score, "sd": None,
            "fit_episodes": len(train_y), "eval_episodes": len(val_y),
            "chance": 1.0 / classes, "feature_dim": train_x.shape[1],
            "readout": ("train-bank StandardScaler+LogisticRegression"
                        "(C=1,lbfgs,random_state=0); fixed val bank")}


def probe_categorical(train_data: dict, train_prior: np.ndarray,
                      val_data: dict, val_prior: np.ndarray) -> dict:
    train_x = categorical_features(train_data, train_prior)
    val_x = categorical_features(val_data, val_prior)
    train_y = np.asarray(train_data["xi"], dtype=np.int64)
    val_y = np.asarray(val_data["xi"], dtype=np.int64)
    train_contract = categorical_probe_contract(train_data)
    val_contract = categorical_probe_contract(val_data)
    if train_contract != val_contract:
        raise ValueError("train/validation decision endpoint contracts differ")
    result = _fit_categorical_probe(train_x, train_y, val_x, val_y)
    result.update({
        "role": "primary_registered_decision_endpoint",
        "feature": "concat(flatten(z[:, q-H:q]), prior_read[:, q])",
        "endpoint_contract": train_contract,
    })
    return result


def probe_categorical_trajectory(train_data: dict, train_prior: np.ndarray,
                                 val_data: dict, val_prior: np.ndarray) -> dict:
    """Exploratory post-cue state summary, separated from the main endpoint."""

    train_x = categorical_trajectory_features(train_data, train_prior)
    val_x = categorical_trajectory_features(val_data, val_prior)
    train_y = np.asarray(train_data["xi"], dtype=np.int64)
    val_y = np.asarray(val_data["xi"], dtype=np.int64)
    endpoint = categorical_probe_contract(train_data)
    if endpoint != categorical_probe_contract(val_data):
        raise ValueError("train/validation trajectory endpoint contracts differ")
    result = _fit_categorical_probe(train_x, train_y, val_x, val_y)
    result.update({
        "role": "exploratory_secondary_trajectory_probe",
        "feature": ("concat(flatten(z[:, q-H:q]), "
                    "mean(prior_read[:, cue_off+2:q+1]), prior_read[:, q])"),
        "endpoint_contract": {
            "schema": TRAJECTORY_PROBE_SCHEMA,
            "decision_observation_index": endpoint["decision_observation_index"],
            "raw_context_indices": endpoint["raw_context_indices"],
            "current_observation_excluded": True,
            "future_observation_consumed": False,
            "temporal_aggregation": True,
            "aggregation": "mean prior_read[t] for cue_off+2 <= t <= q",
            "prior_timing": "every prior_read[:, t] precedes z[:, t]",
        },
    })
    return result


def continuous_features(data: dict, prior: np.ndarray) -> np.ndarray:
    z = np.asarray(data["z"], dtype=np.float64)
    gap_off = np.asarray(data["event_gap_off"], dtype=np.int64)
    features = []
    for episode, time in enumerate(gap_off):
        context = z[episode, time - OFFICIAL_HISTORY:time].reshape(-1)
        features.append(np.concatenate([context, prior[episode, time]]))
    return np.asarray(features)


def probe_continuous(train_data: dict, train_prior: np.ndarray,
                     val_data: dict, val_prior: np.ndarray) -> dict:
    train_x = continuous_features(train_data, train_prior)
    val_x = continuous_features(val_data, val_prior)
    train_y = np.asarray(train_data["xi"], dtype=np.float64)
    val_y = np.asarray(val_data["xi"], dtype=np.float64)
    y_scaler = StandardScaler().fit(train_y)
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
    model.fit(train_x, y_scaler.transform(train_y))
    prediction = y_scaler.inverse_transform(model.predict(val_x))
    score = float(r2_score(val_y, prediction))
    return {"metric": "r2", "mean": score, "sd": None,
            "fit_episodes": len(train_y), "eval_episodes": len(val_y),
            "chance": 0.0, "feature_dim": train_x.shape[1],
            "readout": "train-bank standardized-X/Y RidgeCV; fixed val bank"}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--arm", required=True, choices=FROZEN_CARRIER_NAMES)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--cache-root",
                        default="outputs/paper_a_expansion/cache")
    parser.add_argument("--weights", default=(
        "outputs/paper_a_expansion/pretrained/lewm-reacher/weights.pt"))
    parser.add_argument("--output",
                        default="outputs/paper_a_expansion/frozen_swap")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--study",
                        default="official-lewm-frozen-carrier-swap")
    parser.add_argument(
        "--provenance-spec", type=Path,
        help=("locked strengthening spec; when supplied, task/arm/seed, "
              "inputs, hyperparameters, and output root must match it"))
    return parser.parse_args(argv)


def strengthening_provenance(args: argparse.Namespace, train_path: Path,
                             validation_path: Path) -> dict | None:
    if args.provenance_spec is None:
        return None
    spec = load_locked_spec(args.provenance_spec)
    validate_device(spec, args.device)
    wave = spec["carrier_seed_extension"]
    if args.task not in spec["tasks"] or args.arm not in wave["arms"] \
            or args.seed not in wave["seeds"]:
        raise ValueError(
            "requested carrier cell is outside the locked seed-extension grid")
    expected = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
    }
    for key, actual in expected.items():
        if wave[key] != actual:
            raise ValueError(
                f"seed-extension {key} differs from locked spec: "
                f"{actual!r} != {wave[key]!r}")
    if args.study != "official-lewm-frozen-carrier-seed-extension-v1":
        raise ValueError("strengthening run uses the wrong study identity")
    expected_output = resolve_spec_path(
        spec, spec["output"]["carrier_seed_extension"])
    if Path(args.output).resolve() != expected_output:
        raise ValueError("strengthening output root differs from locked spec")
    expected_train = spec["parent"]["train_caches"][args.task]
    expected_validation = spec["parent"]["validation_caches"][args.task]
    if train_path.resolve() != resolve_spec_path(spec, expected_train["path"]) \
            or validation_path.resolve() != resolve_spec_path(
                spec, expected_validation["path"]):
        raise ValueError("strengthening caches differ from locked parent inputs")
    weights = resolve_spec_path(spec, spec["parent"]["official_weights"]["path"])
    if Path(args.weights).resolve() != weights:
        raise ValueError("strengthening checkpoint differs from locked official host")
    return spec["_spec_record"]


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    cache_root = Path(args.cache_root) / args.task
    train_path = cache_root / "train.npz"
    validation_path = cache_root / "val.npz"
    provenance = strengthening_provenance(
        args, train_path, validation_path)
    train = load_cache(train_path)
    val = load_cache(validation_path)
    train_z = np.asarray(train["z"], dtype=np.float32)
    train_actions = np.asarray(train["actions"], dtype=np.float32)
    val_z = np.asarray(val["z"], dtype=np.float32)
    val_actions = np.asarray(val["actions"], dtype=np.float32)

    model = load_official_reacher_checkpoint(args.weights, device)
    model.eval()
    before = state_digest(model)
    carrier = make_frozen_carrier(
        args.arm, OFFICIAL_EMBED_DIM, OFFICIAL_ACTION_DIM).to(device)
    output_dir = Path(args.output) / args.task / args.arm / f"s{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("history.csv", "metrics.json", "carrier.pt",
                     "eval_export.npz"):
        if (output_dir / filename).exists():
            raise FileExistsError(f"refusing to overwrite {output_dir / filename}")

    rows = []
    if carrier.parameter_count() > 0:
        optimizer = torch.optim.AdamW(carrier.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs, 1))
        rng = np.random.default_rng(61_000 + args.seed)
        with (output_dir / "history.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream,
                                    fieldnames=("epoch", "loss", "lr"))
            writer.writeheader()
            for epoch in range(1, args.epochs + 1):
                loss = train_epoch(model, carrier, optimizer, train_z,
                                   train_actions, args.batch_size, rng, device)
                scheduler.step()
                row = {"epoch": epoch, "loss": loss,
                       "lr": optimizer.param_groups[0]["lr"]}
                rows.append(row)
                writer.writerow(row)
                stream.flush()
                if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                    print(f"[frozen-swap] {args.task}/{args.arm}/s{args.seed} "
                          f"e{epoch}/{args.epochs} loss={loss:.5f}", flush=True)
    else:
        with (output_dir / "history.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("epoch", "loss", "lr"))
            writer.writeheader()

    _, train_prior = carrier_outputs(
        carrier, train_z, train_actions, device)
    fused, prior = carrier_outputs(carrier, val_z, val_actions, device)
    val_mse = prediction_mse(model, fused, val_actions, val_z, device)
    categorical = np.asarray(val["xi"]).ndim == 1
    probe = (probe_categorical(train, train_prior, val, prior)
             if categorical
             else probe_continuous(train, train_prior, val, prior))
    trajectory_probe = (probe_categorical_trajectory(
        train, train_prior, val, prior) if categorical else None)
    after = state_digest(model)
    if before != after:
        raise RuntimeError("frozen official LeWM host changed during carrier training")
    metrics = {
        "schema_version": 1,
        "study": args.study,
        "task": args.task,
        "arm": args.arm,
        "seed": args.seed,
        "official_host": "quentinll/lewm-reacher",
        "official_host_state_sha256_before": before,
        "official_host_state_sha256_after": after,
        "frozen_host_unchanged": True,
        "host_trainable_parameters": 0,
        "carrier_parameters": carrier.parameter_count(),
        "parameter_matching": parameter_report(
            OFFICIAL_EMBED_DIM, OFFICIAL_ACTION_DIM),
        "carrier_config": carrier.describe(),
        "epochs": args.epochs if carrier.parameter_count() else 0,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "final_train_loss": rows[-1]["loss"] if rows else None,
        "val_next_latent_mse": val_mse,
        "probe": probe,
        "source_caches": {
            "train": {"path": str(train_path),
                      "sha256": sha256_file(train_path)},
            "validation": {"path": str(validation_path),
                           "sha256": sha256_file(validation_path)},
        },
    }
    if provenance is not None:
        metrics["strengthening_spec"] = provenance
    if trajectory_probe is not None:
        metrics["trajectory_probe"] = trajectory_probe
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    torch.save({"carrier_state_dict": carrier.state_dict(),
                "metrics": metrics}, output_dir / "carrier.pt")
    export = {"prior_read": prior, "z": val_z, "actions": val_actions,
              "xi": np.asarray(val["xi"]),
              "meta_json": np.array(json.dumps({
                  "task": args.task, "arm": args.arm, "seed": args.seed,
                  "host": "official_frozen_sigreg_lewm"}, sort_keys=True))}
    for key, value in val.items():
        if key.startswith("event_"):
            export[key] = value
    np.savez_compressed(output_dir / "eval_export.npz", **export)
    print(f"[frozen-swap] wrote {output_dir}; {probe['metric']}="
          f"{probe['mean']:.3f}, host unchanged", flush=True)


if __name__ == "__main__":
    main()
