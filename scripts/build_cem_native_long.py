#!/usr/bin/env python3
"""Build label-free native long-trajectory DINO features and query recipes."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_cem_raw_ogbench import (  # noqa: E402
    encode_dino_patch_pyramid,
    env_family,
    json_safe,
    load_dinov2,
    load_raw_cache,
    resolve_device,
    sha256_file,
    split_indices,
    stable_json,
)


DEFAULT_OUTPUT = ROOT / "outputs/cem_native_long_v1"
DEFAULT_CACHE_ROOT = ROOT / "outputs/paper_c_agescale_v1/cache"
DEFAULT_DINOV2 = (
    ROOT / "outputs/dinowm_native_pusht_audit_v1/vendor/dinov2"
)
DEFAULT_TORCH_HOME = (
    ROOT / "outputs/dinowm_native_pusht_audit_v1/torch_home"
)
DEFAULT_DINO_WEIGHTS = (
    DEFAULT_TORCH_HOME / "hub/checkpoints/dinov2_vits14_pretrain.pth"
)
ENVIRONMENTS = (
    "pointmaze-large-navigate-v0",
    "cube-single-play-v0",
    "puzzle-3x3-play-v0",
    "scene-play-v0",
)
GAPS = (16, 32, 64, 128)
CONTEXT = 4
HORIZON = 4
RECENT_TOKENS = 4
FORBIDDEN_CALLS = {
    "draw_cue",
    "inject_cue_sequence",
    "inject_cue_sequence_mode",
    "_saliency_map",
}


def feature_path(output: Path, env_name: str) -> Path:
    return output / "features" / env_name / "features.npz"


def recipe_path(output: Path, env_name: str) -> Path:
    return output / "build" / env_name / "queries.npz"


def source_audit() -> dict[str, Any]:
    tree = ast.parse(Path(__file__).read_text())
    call_sites = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name in FORBIDDEN_CALLS:
            call_sites.append({"name": name, "line": int(node.lineno)})
    return {
        "passed": not call_sites,
        "forbidden_call_sites": call_sites,
        "consumed_cache_arrays": ["frames", "actions"],
        "ignored_cache_arrays": [
            "cue_labels",
            "cue_positions",
            "rewards",
            "goal_state",
            "simulator_state",
        ],
        "synthetic_cue_injection": False,
        "cue_labels_or_times_consumed": False,
        "manual_event_labels": False,
        "manual_saliency": False,
        "frames_modified_before_encoding": False,
        "query_mining_inputs": [
            "frozen DINO features",
            "actions",
            "timestamps",
            "trajectory split",
        ],
    }


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def prepare_features(args: argparse.Namespace) -> dict[str, Any]:
    """Reuse the raw CEM encoder and fit the projection on train episodes only."""

    from sklearn.decomposition import PCA
    import torch

    path = feature_path(args.output, args.env_name)
    receipt_path = path.parent / "receipt.json"
    if path.is_file() and receipt_path.is_file() and not args.overwrite:
        return json.loads(receipt_path.read_text())
    path.parent.mkdir(parents=True, exist_ok=True)
    frames, actions, raw_receipt = load_raw_cache(
        args.cache_root,
        args.env_name,
        args.max_episodes,
    )
    if frames.shape[1] < max(GAPS) + HORIZON + 1:
        raise ValueError(
            f"{args.env_name} has only {frames.shape[1]} frames; "
            f"at least {max(GAPS) + HORIZON + 1} are required"
        )
    train_idx, val_idx, test_idx = split_indices(len(frames))
    if min(map(len, (train_idx, val_idx, test_idx))) < 2:
        raise ValueError("each trajectory split must contain at least two episodes")
    device = resolve_device(args.gpu)
    model = load_dinov2(args.dinov2, args.torch_home, device)
    started = time.time()
    pyramid = encode_dino_patch_pyramid(
        model,
        frames,
        device,
        args.feature_batch_size,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    raw_dim = int(pyramid.shape[-1])
    components = min(
        args.latent_dim,
        raw_dim,
        int(len(train_idx) * frames.shape[1] - 1),
    )
    projection = PCA(
        n_components=components,
        svd_solver="randomized",
        iterated_power=3,
        random_state=20_260_723,
    )
    train_flat = pyramid[train_idx].reshape(-1, raw_dim).astype(np.float32)
    projection.fit(train_flat)
    latent = projection.transform(
        pyramid.reshape(-1, raw_dim).astype(np.float32)
    ).reshape(len(frames), frames.shape[1], components)
    scale = latent[train_idx].reshape(-1, components).std(0)
    scale = np.maximum(scale, 1e-4)
    latent = (latent / scale).astype(np.float32)
    np.savez_compressed(
        path,
        latents=latent,
        actions=actions.astype(np.float32),
        train_indices=train_idx.astype(np.int64),
        val_indices=val_idx.astype(np.int64),
        test_indices=test_idx.astype(np.int64),
        pca_mean=projection.mean_.astype(np.float32),
        pca_components=projection.components_.astype(np.float32),
        latent_scale=scale.astype(np.float32),
    )
    audit = source_audit()
    if not audit["passed"]:
        raise RuntimeError(f"native source contract failed: {audit}")
    receipt = {
        "schema": "cem_native_long_feature_receipt_v1",
        "status": "completed",
        "environment": args.env_name,
        "family": env_family(args.env_name),
        "trajectory_frames": int(frames.shape[1]),
        "trajectory_actions": int(actions.shape[1]),
        "episode_count": int(len(frames)),
        "native_chronology": True,
        "controlled_splicing": False,
        "raw_cache": raw_receipt,
        "split": {
            "trajectory_disjoint": True,
            "train_count": int(len(train_idx)),
            "validation_count": int(len(val_idx)),
            "test_count": int(len(test_idx)),
            "train_indices_sha256": _digest(train_idx),
            "validation_indices_sha256": _digest(val_idx),
            "test_indices_sha256": _digest(test_idx),
        },
        "semantic_encoder": {
            "name": "DINOv2 ViT-S/14",
            "frozen": True,
            "weights": str(args.dino_weights.relative_to(ROOT)),
            "weights_sha256": sha256_file(args.dino_weights),
            "spatial_reduction": "1x1 plus 2x2 patch-token pyramid",
            "raw_dim": raw_dim,
            "train_only_pca_dim": int(components),
        },
        "source_contract": audit,
        "feature_path": str(path.relative_to(ROOT)),
        "feature_sha256": sha256_file(path),
        "elapsed_seconds": float(time.time() - started),
    }
    receipt_path.write_text(stable_json(json_safe(receipt)))
    return receipt


def _cosine_rows(values: np.ndarray, query: np.ndarray) -> np.ndarray:
    numerator = values @ query
    denominator = np.linalg.norm(values, axis=1) * np.linalg.norm(query)
    return numerator / np.maximum(denominator, 1e-8)


def _query_records(
    latents: np.ndarray,
    actions: np.ndarray,
    episodes: np.ndarray,
    gap: int,
) -> list[dict[str, float | int]]:
    change = np.zeros(latents.shape[:2], dtype=np.float32)
    change[:, 1:] = np.mean(
        np.square(latents[:, 1:] - latents[:, :-1]),
        axis=-1,
    )
    action_change = np.zeros(actions.shape[:2], dtype=np.float32)
    action_change[:, 1:] = np.mean(
        np.square(actions[:, 1:] - actions[:, :-1]),
        axis=-1,
    )
    records: list[dict[str, float | int]] = []
    maximum_query = latents.shape[1] - HORIZON - 1
    minimum_query = max(gap, CONTEXT + RECENT_TOKENS - 1)
    for episode_value in episodes:
        episode = int(episode_value)
        for query_t in range(minimum_query, maximum_query + 1):
            old_stop = query_t - gap
            if old_stop < 0:
                continue
            current = latents[
                episode,
                query_t - CONTEXT + 1 : query_t + 1,
            ].mean(0)
            old = latents[episode, : old_stop + 1]
            similarity = _cosine_rows(old, current)
            revisit_index = int(np.argmax(similarity))
            recent_stop = query_t - CONTEXT
            middle = latents[
                episode,
                revisit_index + 1 : max(revisit_index + 2, recent_stop),
            ]
            middle_similarity = (
                float(_cosine_rows(middle, current).mean())
                if len(middle)
                else float(similarity[revisit_index])
            )
            recent_start = recent_stop - RECENT_TOKENS + 1
            recent_mean = latents[
                episode,
                recent_start : recent_stop + 1,
            ].mean(0)
            records.append(
                {
                    "episode_id": episode,
                    "query_t": query_t,
                    "gap": gap,
                    "revisit": float(similarity[revisit_index]),
                    "reappearance": float(
                        similarity[revisit_index] - middle_similarity
                    ),
                    "transition": float(change[episode, query_t]),
                    "action_structure": float(
                        action_change[episode, min(query_t - 1, actions.shape[1] - 1)]
                    ),
                    "region_change": float(
                        np.mean(np.square(current - recent_mean))
                    ),
                }
            )
    return records


def _fit_signal_scale(
    records: list[dict[str, float | int]],
) -> dict[str, tuple[float, float]]:
    keys = (
        "revisit",
        "reappearance",
        "transition",
        "action_structure",
        "region_change",
    )
    scale = {}
    for key in keys:
        values = np.asarray([row[key] for row in records], dtype=np.float64)
        median = float(np.median(values))
        spread = float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
        scale[key] = (median, max(spread, 1e-8))
    return scale


def _score_records(
    records: list[dict[str, float | int]],
    scale: dict[str, tuple[float, float]],
) -> None:
    for row in records:
        standardized = [
            np.clip(
                (float(row[key]) - scale[key][0]) / scale[key][1],
                -4.0,
                4.0,
            )
            for key in scale
        ]
        row["proposal_score"] = float(np.mean(standardized))


def _select_queries(
    records: list[dict[str, float | int]],
    count: int,
) -> list[dict[str, float | int]]:
    grouped: dict[int, list[dict[str, float | int]]] = {}
    for row in records:
        grouped.setdefault(int(row["episode_id"]), []).append(row)
    selected = []
    for rows in grouped.values():
        ranked = sorted(
            rows,
            key=lambda row: float(row["proposal_score"]),
            reverse=True,
        )
        keep: list[dict[str, float | int]] = []
        for row in ranked:
            if all(
                abs(int(row["query_t"]) - int(other["query_t"])) >= HORIZON
                for other in keep
            ):
                keep.append(row)
            if len(keep) == count:
                break
        if len(keep) < count:
            for row in ranked:
                if row not in keep:
                    keep.append(row)
                if len(keep) == count:
                    break
        selected.extend(sorted(keep, key=lambda row: int(row["query_t"])))
    return selected


def build_queries(args: argparse.Namespace) -> dict[str, Any]:
    path = feature_path(args.output, args.env_name)
    if not path.is_file():
        raise FileNotFoundError(f"prepare long features first: {path}")
    with np.load(path, allow_pickle=False) as data:
        latents = np.asarray(data["latents"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        splits = {
            "train": np.asarray(data["train_indices"], dtype=np.int64),
            "validation": np.asarray(data["val_indices"], dtype=np.int64),
            "test": np.asarray(data["test_indices"], dtype=np.int64),
        }
    split_sets = {name: set(value.tolist()) for name, value in splits.items()}
    if (
        split_sets["train"] & split_sets["validation"]
        or split_sets["train"] & split_sets["test"]
        or split_sets["validation"] & split_sets["test"]
    ):
        raise RuntimeError("trajectory split leakage")
    arrays: dict[str, np.ndarray] = {
        "gaps": np.asarray(GAPS, dtype=np.int64),
    }
    summaries: dict[str, Any] = {}
    signal_scales: dict[str, Any] = {}
    for gap in GAPS:
        records = {
            name: _query_records(latents, actions, indices, gap)
            for name, indices in splits.items()
        }
        scale = _fit_signal_scale(records["train"])
        signal_scales[str(gap)] = {
            key: {"median": value[0], "iqr": value[1]}
            for key, value in scale.items()
        }
        for rows in records.values():
            _score_records(rows, scale)
        for split, rows in records.items():
            chosen = _select_queries(rows, args.queries_per_episode)
            prefix = f"{split}_g{gap}_"
            for key in (
                "episode_id",
                "query_t",
                "gap",
                "revisit",
                "reappearance",
                "transition",
                "action_structure",
                "region_change",
                "proposal_score",
            ):
                dtype = np.int64 if key in {"episode_id", "query_t", "gap"} else np.float32
                arrays[prefix + key] = np.asarray(
                    [row[key] for row in chosen],
                    dtype=dtype,
                )
            summaries.setdefault(split, {})[str(gap)] = {
                "eligible_query_count": int(len(rows)),
                "selected_query_count": int(len(chosen)),
                "episode_count": int(len(splits[split])),
                "queries_per_episode": args.queries_per_episode,
            }
    output = recipe_path(args.output, args.env_name)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    audit = source_audit()
    if not audit["passed"]:
        raise RuntimeError(f"native source contract failed: {audit}")
    receipt = {
        "schema": "cem_native_long_query_build_v1",
        "status": "completed",
        "environment": args.env_name,
        "family": env_family(args.env_name),
        "native_chronology": True,
        "controlled_splicing": False,
        "trajectory_frames": int(latents.shape[1]),
        "gaps": list(GAPS),
        "prediction_horizon": HORIZON,
        "host_context": CONTEXT,
        "query_mining": (
            "train-scaled frozen-DINO revisit/reappearance, semantic transition, "
            "action transition, and recent-region change; top causal-prefix "
            "queries per episode"
        ),
        "future_target_used_for_query_mining": False,
        "signal_scales_fit_on_train_only": signal_scales,
        "splits": summaries,
        "split_leakage": False,
        "source_contract": audit,
        "artifacts": {
            "queries": str(output.relative_to(ROOT)),
            "features": str(path.relative_to(ROOT)),
        },
    }
    (output.parent / "receipt.json").write_text(
        stable_json(json_safe(receipt))
    )
    return receipt


def build_environment(args: argparse.Namespace) -> dict[str, Any]:
    feature_receipt = prepare_features(args)
    query_receipt = build_queries(args)
    result = {
        "schema": "cem_native_long_build_environment_v1",
        "status": "completed",
        "environment": args.env_name,
        "feature": feature_receipt,
        "queries": query_receipt,
    }
    print(stable_json(json_safe(result)), flush=True)
    return result


def aggregate_build(output: Path) -> dict[str, Any]:
    receipts = []
    for path in sorted((output / "build").glob("*/receipt.json")):
        receipts.append(json.loads(path.read_text()))
    report = {
        "schema": "cem_native_long_build_report_v1",
        "status": "completed" if receipts else "empty",
        "environment_count": len(receipts),
        "environments": [row["environment"] for row in receipts],
        "families": sorted({row["family"] for row in receipts}),
        "gaps": list(GAPS),
        "all_native_chronology": all(
            row["native_chronology"] for row in receipts
        ),
        "any_controlled_splicing": any(
            row["controlled_splicing"] for row in receipts
        ),
        "all_source_contracts_pass": all(
            row["source_contract"]["passed"] for row in receipts
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "build_report.json").write_text(stable_json(report))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-name")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--dinov2", type=Path, default=DEFAULT_DINOV2)
    parser.add_argument("--torch-home", type=Path, default=DEFAULT_TORCH_HOME)
    parser.add_argument("--dino-weights", type=Path, default=DEFAULT_DINO_WEIGHTS)
    parser.add_argument("--latent-dim", type=int, default=96)
    parser.add_argument("--feature-batch-size", type=int, default=384)
    parser.add_argument("--queries-per-episode", type=int, default=2)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    for name in (
        "output",
        "cache_root",
        "dinov2",
        "torch_home",
        "dino_weights",
    ):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    if args.gpu == 3 or args.gpu not in (0, 1, 2):
        parser.error("--gpu must be one of 0,1,2; GPU3 is prohibited")
    if not args.aggregate and args.all == bool(args.env_name):
        parser.error("choose exactly one of --all or --env-name")
    if args.env_name and args.env_name not in ENVIRONMENTS:
        parser.error(f"unsupported environment: {args.env_name}")
    if args.smoke:
        args.max_episodes = args.max_episodes or 48
        args.latent_dim = min(args.latent_dim, 32)
        args.queries_per_episode = 1
        args.feature_batch_size = min(args.feature_batch_size, 256)
    return args


def main() -> None:
    args = parse_args()
    if args.aggregate:
        print(stable_json(aggregate_build(args.output)), flush=True)
        return
    environments = ENVIRONMENTS if args.all else (args.env_name,)
    results = []
    for environment in environments:
        args.env_name = environment
        results.append(build_environment(args))
    if args.all:
        print(stable_json(aggregate_build(args.output)), flush=True)


if __name__ == "__main__":
    main()
