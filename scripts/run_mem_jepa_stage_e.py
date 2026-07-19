#!/usr/bin/env python3
"""Stage-E executed-use gate for label-free Mem-JEPA on DINO-WM PointMaze."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from lewm.official_tasks.dinowm_pointmaze import crossed_execution_arrays  # noqa: E402
from scripts.run_mem_jepa_stage_b import (  # noqa: E402
    DEFAULT_CONFIG,
    FeatureBank,
    FrozenPointMazeHost,
    atomic_json,
    load_config,
    require,
    resolve,
    set_determinism,
    sha256_file,
)
from scripts.run_mem_jepa_stage_c import (  # noqa: E402
    MemJepaLabelFreeAdapter,
    collect_features,
    endpoint_frame,
    train_one_age,
)


DEFAULT_OUTPUT = ROOT / "outputs/mem_jepa_stage_e"
ARMS = ("full", "reset", "no_state")
SEEDS = (0, 1, 2, 3, 4)
AGE = 15


def expanded_labels(base_count: int) -> np.ndarray:
    return np.tile(np.arange(4, dtype=np.int64), int(base_count))


def episode_cluster_bootstrap(values: np.ndarray, episodes: np.ndarray, *,
                              draws: int, seed: int,
                              confidence: float = 0.95) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    episodes = np.asarray(episodes, dtype=np.int64)
    require(values.ndim == 2 and values.shape[1] == len(episodes),
            "bootstrap values must be (seed, expanded_example)")
    unique = np.unique(episodes)
    per_episode = np.stack([
        values[:, episodes == episode].mean(axis=1) for episode in unique
    ], axis=1)
    point = float(per_episode.mean())
    rng = np.random.default_rng(seed)
    samples = np.empty(int(draws), dtype=np.float64)
    cursor = 0
    while cursor < int(draws):
        stop = min(int(draws), cursor + 512)
        count = stop - cursor
        seed_rows = rng.integers(0, values.shape[0], size=(count, values.shape[0]))
        episode_rows = rng.integers(0, len(unique), size=(count, len(unique)))
        selected = per_episode[seed_rows[:, :, None], episode_rows[:, None, :]]
        samples[cursor:stop] = selected.mean(axis=(1, 2))
        cursor = stop
    alpha = (1.0 - float(confidence)) / 2.0
    interval = np.quantile(samples, (alpha, 1.0 - alpha))
    return {
        "mean": point,
        "ci95": interval.astype(float).tolist(),
        "draws": int(draws),
        "seed": int(seed),
        "confidence": float(confidence),
        "paired": True,
        "equal_native_episode_weight": True,
        "native_episode_clusters": int(len(unique)),
        "carrier_seeds": int(values.shape[0]),
        "ci_excludes_zero": bool(interval[0] > 0 or interval[1] < 0),
    }


def train_seed(args: argparse.Namespace) -> dict[str, Any]:
    output = resolve(args.output)
    seed_dir = output / "seeds" / f"s{args.seed_index}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(resolve(args.config))
    run_seed = int(args.seed_base + 100 * args.seed_index + AGE)
    set_determinism(run_seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    bank = FeatureBank(cfg)
    host = FrozenPointMazeHost(cfg, device)
    host_before = host.digest()
    model = MemJepaLabelFreeAdapter(
        dim=args.dim, slots=args.slots, heads=args.heads).to(device)
    started = time.time()
    history = train_one_age(
        model, host, bank, age=AGE, seed=run_seed, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
        temperature=args.temperature, variant="full", output_dir=seed_dir)

    train_truth = expanded_labels(len(bank.base_indices("train")))
    validation_truth = expanded_labels(len(bank.base_indices("validation")))
    arrays: dict[str, np.ndarray] = {
        "train_truth": train_truth,
        "validation_truth": validation_truth,
    }
    for split in ("train", "validation"):
        for arm in ARMS:
            pack = collect_features(
                model, host, bank, split=split, age=AGE, condition=arm,
                batch_size=args.eval_batch_size)
            arrays[f"{split}_{arm}_feature"] = pack["features"].astype(np.float32)
            arrays[f"{split}_{arm}_retrieval"] = pack[
                "retrieval_prediction"].astype(np.int64)
    feature_path = seed_dir / "use_features.npz"
    np.savez_compressed(feature_path, **arrays)
    host_after = host.digest()
    require(host_before == host_after, "frozen host digest changed")
    result = {
        "schema": "mem_jepa_stage_e_seed_v1",
        "status": "completed",
        "seed_index": int(args.seed_index),
        "run_seed": run_seed,
        "age": AGE,
        "endpoint_frame": endpoint_frame(3, AGE),
        "labels_used_for_adapter_training": False,
        "labels_used_for_consumer_training": True,
        "host_digest_unchanged": True,
        "feature_path": str(feature_path.relative_to(ROOT)),
        "feature_sha256": sha256_file(feature_path),
        "feature_dim": int(arrays["train_full_feature"].shape[1]),
        "training": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "final": history[-1] if history else None,
        },
        "elapsed_seconds": float(time.time() - started),
    }
    atomic_json(seed_dir / "seed_summary.json", result)
    print(json.dumps({
        "seed": args.seed_index,
        "feature_dim": result["feature_dim"],
        "elapsed": round(result["elapsed_seconds"], 1),
    }, indent=2), flush=True)
    return result


def load_seed_features(output: Path, seed_index: int) -> dict[str, np.ndarray]:
    path = output / "seeds" / f"s{seed_index}" / "use_features.npz"
    require(path.is_file(), f"missing seed features: {path}")
    with np.load(path) as values:
        return {name: values[name] for name in values.files}


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    output = resolve(args.output)
    cfg = load_config(resolve(args.config))
    bank = FeatureBank(cfg)
    deck_path = bank.root / "execution_deck.npz"
    require(deck_path.is_file(), f"missing execution deck: {deck_path}")
    with np.load(deck_path) as deck:
        success_matrix = deck["success_matrix"]
        validation_episode = deck["validation_episode"]
    episodes = np.repeat(validation_episode, 4)
    seeds = list(map(int, args.seeds))
    truth = None
    predictions: dict[str, list[np.ndarray]] = {arm: [] for arm in ARMS}
    consumer_receipts = []
    for seed in seeds:
        features = load_seed_features(output, seed)
        if truth is None:
            truth = features["validation_truth"].astype(np.int64)
        train_x = np.concatenate([features[f"train_{arm}_feature"] for arm in ARMS])
        train_y = np.concatenate([features["train_truth"] for _ in ARMS])
        classifier = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0))
        classifier.fit(train_x, train_y)
        digest = hashlib.sha256(
            classifier[-1].coef_.astype(np.float64).tobytes()).hexdigest()
        for arm in ARMS:
            require(np.array_equal(features["validation_truth"], truth),
                    "validation truth changed across seeds")
            predictions[arm].append(classifier.predict(
                features[f"validation_{arm}_feature"]).astype(np.int64))
        consumer_receipts.append({
            "seed_index": int(seed),
            "consumer": "StandardScaler+RidgeClassifier(alpha=1)",
            "arm_blind": True,
            "training_arms": list(ARMS),
            "train_examples": int(len(train_y)),
            "feature_dim": int(train_x.shape[1]),
            "coefficient_sha256": digest,
        })
    assert truth is not None
    prediction_matrices = {
        arm: np.stack(values).astype(np.int64)
        for arm, values in predictions.items()
    }
    executed, goal_correct = {}, {}
    for arm in ARMS:
        arm_executed, arm_correct = [], []
        for seed_index in range(len(seeds)):
            crossed = crossed_execution_arrays(
                success_matrix, prediction_matrices[arm][seed_index], truth)
            arm_executed.append(crossed["executed_success"])
            arm_correct.append(crossed["goal_correct"])
        executed[arm] = np.stack(arm_executed).astype(np.float64)
        goal_correct[arm] = np.stack(arm_correct).astype(np.float64)

    random_predictions, random_executed = [], []
    for seed in seeds:
        rng = np.random.default_rng(int(args.random_seed) + int(seed))
        prediction = rng.integers(0, 4, size=len(truth), dtype=np.int64)
        random_predictions.append(prediction)
        random_executed.append(crossed_execution_arrays(
            success_matrix, prediction, truth)["executed_success"])
    random_executed_matrix = np.stack(random_executed).astype(np.float64)

    arm_results = {}
    for index, arm in enumerate(ARMS):
        result = {
            "goal_accuracy": episode_cluster_bootstrap(
                goal_correct[arm], episodes, draws=args.draws,
                seed=args.bootstrap_seed + 100 + index),
            "executed_success": episode_cluster_bootstrap(
                executed[arm], episodes, draws=args.draws,
                seed=args.bootstrap_seed + 200 + index),
            "contrast_vs_no_state": episode_cluster_bootstrap(
                executed[arm] - executed["no_state"], episodes,
                draws=args.draws, seed=args.bootstrap_seed + 300 + index),
            "contrast_vs_random": episode_cluster_bootstrap(
                executed[arm] - random_executed_matrix, episodes,
                draws=args.draws, seed=args.bootstrap_seed + 400 + index),
        }
        result["resolved_execution_gain"] = bool(
            arm == "full"
            and result["contrast_vs_no_state"]["ci95"][0] > 0
            and result["contrast_vs_random"]["ci95"][0] > 0)
        arm_results[arm] = result
    random_result = episode_cluster_bootstrap(
        random_executed_matrix, episodes, draws=args.draws,
        seed=args.bootstrap_seed + 500)
    prediction_path = output / "execution_predictions.npz"
    np.savez_compressed(
        prediction_path, truth=truth, validation_episode=episodes,
        success_matrix=success_matrix,
        random_prediction=np.stack(random_predictions),
        random_executed_success=random_executed_matrix,
        **{f"prediction__{arm}": value for arm, value in prediction_matrices.items()},
        **{f"executed__{arm}": value for arm, value in executed.items()},
    )
    summary = {
        "schema": "mem_jepa_stage_e_summary_v1",
        "status": "completed",
        "age": AGE,
        "labels_used_for_adapter_training": False,
        "labels_used_for_consumer_training": True,
        "execution_deck": str(deck_path.relative_to(ROOT)),
        "consumer": "shared arm-blind StandardScaler+RidgeClassifier(alpha=1)",
        "feature": "compact 1x1+2x2 frozen-host visual pool",
        "claim_boundary": (
            "External memory-conditioned goal selection plus released waypoint "
            "controller; not native DINO-WM planning."),
        "seeds": seeds,
        "consumer_receipts": consumer_receipts,
        "arms": arm_results,
        "realized_random_goal": random_result,
        "artifact": {
            "path": str(prediction_path.relative_to(ROOT)),
            "sha256": sha256_file(prediction_path),
            "size": prediction_path.stat().st_size,
        },
        "resolved_full_execution_gain": bool(
            arm_results["full"]["resolved_execution_gain"]),
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps({
        "resolved_full_execution_gain":
            summary["resolved_full_execution_gain"],
        "full_executed_success":
            summary["arms"]["full"]["executed_success"]["mean"],
        "no_state_executed_success":
            summary["arms"]["no_state"]["executed_success"]["mean"],
        "random_executed_success": summary["realized_random_goal"]["mean"],
    }, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed-index", type=int, choices=SEEDS)
    parser.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--seed-base", type=int, default=9800)
    parser.add_argument("--random-seed", type=int, default=833000)
    parser.add_argument("--bootstrap-seed", type=int, default=984000)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--slots", type=int, default=6)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not args.aggregate and args.seed_index is None:
        parser.error("--seed-index is required unless --aggregate is used")
    return args


def main() -> None:
    args = parse_args()
    if args.aggregate:
        aggregate(args)
    else:
        train_seed(args)


if __name__ == "__main__":
    main()
