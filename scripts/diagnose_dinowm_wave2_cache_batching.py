#!/usr/bin/env python3
"""Quantify the pre-outcome Wave 2 cache batching discrepancy on GPU 1."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lewm.official_tasks.dinowm_native_audit import spatial_pyramid_pool
import scripts.run_dinowm_wave2_spatial_carrier as wave2


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/dinowm_wave2_spatial_carrier.yaml"
OUTPUT = (ROOT / "outputs/dinowm_wave2_spatial_carrier/cache/"
          "numerical_diagnosis.json")
TARGET_INDICES = np.asarray([0, 1, 1199, 1200, 1679], dtype=np.int64)


def stats(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    delta = np.abs(np.asarray(left, dtype=np.float64)
                   - np.asarray(right, dtype=np.float64))
    return {
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "p99_abs": float(np.quantile(delta, 0.99)),
        "p999_abs": float(np.quantile(delta, 0.999)),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
        "exact_fraction": float(np.mean(delta == 0.0)),
    }


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("diagnosis is restricted to physical GPU 1")
    cfg = yaml.safe_load(CONFIG.read_text())
    dataset, selections_by_task = wave2.dataset_and_selections(cfg)
    selections = selections_by_task[cfg["tasks"][0]["key"]]
    cache = np.load(
        ROOT / "outputs/dinowm_wave2_spatial_carrier/cache/base_visual.npy",
        mmap_mode="r")
    with np.load(ROOT / "outputs/dinowm_wave2_spatial_carrier/cache/metadata.npz") as values:
        actions = values["actions"].copy()
        proprio = values["proprio"].copy()
        split = values["split"].copy()
        labels = {task["key"]: values[f"labels__{task['key']}"] .copy()
                  for task in cfg["tasks"]}
    with np.load(ROOT / "outputs/dinowm_native_pusht_audit_v2r2/formal/"
                 "teacher_features.npz") as values:
        preserved = values["predicted_endpoint"].copy()

    host = wave2.FrozenNativeHost(cfg, load_encoder=True)
    host_before = host.digest()
    direct_rows, cache_rows = [], []
    original_batch = 12
    for start in range(0, len(selections), original_batch):
        stop = min(start + original_batch, len(selections))
        native = wave2._read(dataset, selections[start:stop], cfg)
        frames = np.stack([value.frames for value in native])[:, 16:19]
        direct_visual = host.encode_visual(
            frames.reshape(-1, *frames.shape[2:]), batch_size=64).reshape(
                len(native), 3, 196, 384)
        q = torch.from_numpy(proprio[start:stop, 16:19]).to(host.device)
        a = torch.from_numpy(actions[start:stop, 16:19]).to(host.device)
        with torch.inference_mode():
            direct_prediction = host.predict(
                torch.from_numpy(direct_visual).to(host.device), q, a
            )[:, -1, :, :384].float().cpu().numpy()
            cache_prediction = host.predict(
                torch.from_numpy(np.asarray(
                    cache[start:stop, 16:19]).copy()).to(host.device), q, a
            )[:, -1, :, :384].float().cpu().numpy()
        direct_rows.append(spatial_pyramid_pool(direct_prediction))
        cache_rows.append(spatial_pyramid_pool(cache_prediction))
        if stop % 120 == 0:
            print(f"[batch-diagnosis] {stop}/{len(selections)}", flush=True)
    direct = np.concatenate(direct_rows)
    cached = np.concatenate(cache_rows)

    # Reproduce the failed sparse check, where the five selected episodes were
    # also sent through the predictor in a different (B=5) matrix shape.
    q = torch.from_numpy(proprio[TARGET_INDICES, 16:19]).to(host.device)
    a = torch.from_numpy(actions[TARGET_INDICES, 16:19]).to(host.device)
    with torch.inference_mode():
        sparse_prediction = host.predict(
            torch.from_numpy(np.asarray(
                cache[TARGET_INDICES, 16:19]).copy()).to(host.device), q, a
        )[:, -1, :, :384].float().cpu().numpy()
    sparse = spatial_pyramid_pool(sparse_prediction)

    reference64 = preserved.astype(np.float64)
    reference_scale = {
        "min": float(reference64.min()),
        "max": float(reference64.max()),
        "rms": float(np.sqrt(np.mean(np.square(reference64)))),
        "std": float(reference64.std()),
        "mean_abs": float(np.mean(np.abs(reference64))),
        "p99_abs": float(np.quantile(np.abs(reference64), 0.99)),
    }
    row_delta = cached.astype(np.float64) - direct.astype(np.float64)
    relative_l2 = np.linalg.norm(row_delta, axis=1) / np.maximum(
        np.linalg.norm(direct.astype(np.float64), axis=1), 1e-30)
    cosine = np.sum(
        cached.astype(np.float64) * direct.astype(np.float64), axis=1
    ) / np.maximum(
        np.linalg.norm(cached.astype(np.float64), axis=1)
        * np.linalg.norm(direct.astype(np.float64), axis=1), 1e-30)

    decision = {}
    train = split == 0
    validation = split == 1
    for task in cfg["tasks"]:
        key = task["key"]
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0, solver="lbfgs", max_iter=3000, random_state=0))
        model.fit(direct[train], labels[key][train])
        direct_score = model.decision_function(direct[validation])
        cache_score = model.decision_function(cached[validation])
        direct_prediction = np.argmax(direct_score, axis=1)
        cache_prediction = np.argmax(cache_score, axis=1)
        ordered = np.sort(direct_score, axis=1)
        margin = ordered[:, -1] - ordered[:, -2]
        score_delta = np.abs(cache_score - direct_score)
        decision[key] = {
            "classes": int(task["classes"]),
            "direct_balanced_accuracy": float(balanced_accuracy_score(
                labels[key][validation], direct_prediction)),
            "cache_under_direct_readout_balanced_accuracy": float(
                balanced_accuracy_score(
                    labels[key][validation], cache_prediction)),
            "prediction_flips": int(np.count_nonzero(
                direct_prediction != cache_prediction)),
            "validation_episodes": int(validation.sum()),
            "direct_top1_margin_min": float(margin.min()),
            "direct_top1_margin_p01": float(np.quantile(margin, 0.01)),
            "direct_top1_margin_median": float(np.median(margin)),
            "decision_score_delta_max": float(score_delta.max()),
            "decision_score_delta_p99": float(np.quantile(
                score_delta, 0.99)),
        }

    host_after = host.digest()
    if host_before != host_after:
        raise RuntimeError("diagnosis mutated the frozen host")
    result = {
        "schema": "dinowm_wave2_cache_batching_diagnosis_v1",
        "status": "preoutcome_numerical_diagnosis",
        "physical_gpu": 1,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "episodes": len(selections),
        "features_per_episode": int(preserved.shape[1]),
        "preserved_v2r2_teacher_sha256": wave2.sha256_file(
            ROOT / "outputs/dinowm_native_pusht_audit_v2r2/formal/"
            "teacher_features.npz"),
        "sealed_v1_stop_sha256": wave2.sha256_file(
            ROOT / "outputs/dinowm_wave2_spatial_carrier/cache/"
            "stop_receipt.json"),
        "direct_original_batch_vs_preserved_v2r2": stats(direct, preserved),
        "reusable_cache_vs_direct_original_batch": stats(cached, direct),
        "reusable_cache_vs_preserved_v2r2": stats(cached, preserved),
        "failed_sparse_b5_cache_vs_preserved_v2r2": stats(
            sparse, preserved[TARGET_INDICES]),
        "reference_feature_scale": reference_scale,
        "cache_vs_direct_relative_l2": {
            "max": float(relative_l2.max()),
            "mean": float(relative_l2.mean()),
            "p99": float(np.quantile(relative_l2, 0.99)),
        },
        "cache_vs_direct_cosine": {
            "min": float(cosine.min()),
            "mean": float(cosine.mean()),
            "p01": float(np.quantile(cosine, 0.01)),
        },
        "existing_teacher_readout_stability": decision,
        "host_digest_before": host_before,
        "host_digest_after": host_after,
        "host_unchanged": True,
        "carrier_outcomes_computed": False,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    OUTPUT.write_text(encoded)
    print(encoded)
    print("diagnosis_sha256", hashlib.sha256(encoded.encode()).hexdigest())


if __name__ == "__main__":
    main()
