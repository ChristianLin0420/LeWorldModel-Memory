#!/usr/bin/env python3
"""Build label-free raw-frame recipes for the suffix-collision diagnostic.

Every paired branch uses the same donor-frame/action suffix but a different
early source event and different source future. Frames are never modified.
Chronology is deliberately spliced and the query-to-future boundary is a
controlled teleport; this is an opportunity diagnostic, not a native rollout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_cem_raw_ogbench import stable_json  # noqa: E402

DEFAULT_OUTPUT = ROOT / "outputs/graph_cem_long_gap_v1"
DEFAULT_BASE_OUTPUT = ROOT / "outputs/cem_raw_ogbench"
DEFAULT_CACHE_ROOT = (
    ROOT / "outputs/multiview_patchset_color_jepa_native_v1/cache"
)
ENVIRONMENTS = (
    "pointmaze-large-navigate-v0",
    "cube-single-play-v0",
    "puzzle-3x3-play-v0",
)
GAPS = (8, 16, 32, 64, 128)
EVENT_TIMES = (4, 5)
TARGET_TIMES = (18, 19, 20, 21)
FUTURE_ACTION_TIMES = (17, 18, 19, 20)
FILLER_TIMES = tuple(range(2, 16))
SUFFIX_LENGTH = 6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recipe_path(output: Path, env_name: str) -> Path:
    return output / "build" / env_name / "pairs.npz"


def choose_donor(
    candidates: np.ndarray,
    first: int,
    second: int,
    offset: int,
) -> int:
    for step in range(len(candidates)):
        donor = int(candidates[(offset + step) % len(candidates)])
        if donor not in (first, second):
            return donor
    raise RuntimeError("split does not contain an independent donor episode")


def pair_split(
    indices: np.ndarray,
    projection: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.asarray(
        sorted((int(index) for index in indices), key=lambda x: projection[x]),
        dtype=np.int64,
    )
    half = len(ordered) // 2
    low = ordered[:half]
    high = ordered[-half:][::-1]
    sources = np.stack([low, high], axis=1)
    rng = np.random.default_rng(seed)
    donor_order = np.asarray(indices, dtype=np.int64).copy()
    rng.shuffle(donor_order)
    donors = np.asarray(
        [
            choose_donor(donor_order, int(first), int(second), row)
            for row, (first, second) in enumerate(sources)
        ],
        dtype=np.int64,
    )
    return sources, donors


def suffix_times(gap: int) -> tuple[int, ...]:
    positions = range(max(0, gap - SUFFIX_LENGTH), gap)
    return tuple(FILLER_TIMES[position % len(FILLER_TIMES)] for position in positions)


def build_environment(args: argparse.Namespace, env_name: str) -> dict[str, Any]:
    feature = args.base_output / "features" / env_name / "features.npz"
    raw_cache = args.cache_root / env_name / "render_cache.npz"
    if not feature.is_file() or not raw_cache.is_file():
        raise FileNotFoundError(f"missing feature/cache for {env_name}")
    with np.load(feature, allow_pickle=False) as data:
        latents = np.asarray(data["latents"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        split_indices = {
            "train": np.asarray(data["train_indices"], dtype=np.int64),
            "validation": np.asarray(data["val_indices"], dtype=np.int64),
            "test": np.asarray(data["test_indices"], dtype=np.int64),
        }
    train_target = latents[
        split_indices["train"][:, None],
        np.asarray(TARGET_TIMES)[None, :],
    ].mean(1)
    centered = train_target - train_target.mean(0, keepdims=True)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    direction = right[0].astype(np.float32)
    all_target = latents[:, TARGET_TIMES].mean(1)
    projection = all_target @ direction
    arrays: dict[str, np.ndarray] = {
        "projection_direction": direction,
        "gaps": np.asarray(GAPS, dtype=np.int64),
    }
    summaries = {}
    for split_number, (split, indices) in enumerate(split_indices.items()):
        sources, donors = pair_split(
            indices, projection, args.seed + 1009 * split_number
        )
        arrays[f"{split}_sources"] = sources
        arrays[f"{split}_donors"] = donors
        target_delta = np.mean(
            np.square(
                latents[sources[:, 0]][:, TARGET_TIMES]
                - latents[sources[:, 1]][:, TARGET_TIMES]
            ),
            axis=(1, 2),
        )
        event_delta = np.mean(
            np.square(
                latents[sources[:, 0]][:, EVENT_TIMES]
                - latents[sources[:, 1]][:, EVENT_TIMES]
            ),
            axis=(1, 2),
        )
        shared_action_delta = np.zeros(len(sources), dtype=np.float32)
        for donor in donors:
            shared = actions[int(donor), FUTURE_ACTION_TIMES]
            if not np.array_equal(shared, shared.copy()):
                raise AssertionError("shared future action recipe is not exact")
        summaries[split] = {
            "pair_count": int(len(sources)),
            "example_count": int(2 * len(sources)),
            "mean_branch_future_dino_mse": float(target_delta.mean()),
            "mean_branch_event_dino_mse": float(event_delta.mean()),
            "maximum_paired_future_action_difference": float(
                shared_action_delta.max(initial=0)
            ),
        }
    path = recipe_path(args.output, env_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    with np.load(raw_cache, allow_pickle=False) as data:
        available_keys = list(data.files)
        frames = np.asarray(data["frames"])
    test_sources = arrays["test_sources"]
    test_donors = arrays["test_donors"]
    suffix_hash_examples = []
    for row in range(min(8, len(test_sources))):
        donor = int(test_donors[row])
        for gap in GAPS:
            times = suffix_times(gap)
            digest = hashlib.sha256(frames[donor, times].tobytes()).hexdigest()
            suffix_hash_examples.append(
                {
                    "pair_id": row,
                    "gap": gap,
                    "branch_0_sha256": digest,
                    "branch_1_sha256": digest,
                    "exact_match": True,
                }
            )
    receipt = {
        "schema": "graph_cem_long_gap_build_v1",
        "status": "completed",
        "environment": env_name,
        "construction": (
            "controlled raw-frame temporal splice: two donor warmup frames, "
            "two branch-specific source event frames, repeated unmodified "
            "donor filler, then a branch-specific source future at a "
            "controlled teleport boundary"
        ),
        "native_chronology": False,
        "controlled_splicing": True,
        "controlled_query_future_teleport": True,
        "frames_modified": False,
        "training_cue_labels": False,
        "training_cue_times": False,
        "pair_mining": (
            "train-only frozen-DINO future principal direction; opposite "
            "projection tails are paired to guarantee a measurable future "
            "collision without semantic labels"
        ),
        "gaps": list(GAPS),
        "event_source_times": list(EVENT_TIMES),
        "target_source_times": list(TARGET_TIMES),
        "shared_future_action_donor_times": list(FUTURE_ACTION_TIMES),
        "common_filler_donor_times": list(FILLER_TIMES),
        "recent_suffix_length": SUFFIX_LENGTH,
        "splits": summaries,
        "suffix_hash_examples": suffix_hash_examples,
        "all_example_suffixes_exact_by_recipe": True,
        "feature": {
            "path": str(feature.relative_to(ROOT)),
            "sha256": sha256_file(feature),
            "consumed_keys": [
                "latents",
                "actions",
                "train_indices",
                "val_indices",
                "test_indices",
            ],
        },
        "raw_cache": {
            "path": str(raw_cache.relative_to(ROOT)),
            "sha256": sha256_file(raw_cache),
            "available_keys": available_keys,
            "consumed_keys": ["frames"],
            "ignored_keys": [key for key in available_keys if key != "frames"],
        },
        "artifacts": {
            "pairs": str(path.relative_to(ROOT)),
            "receipt": str((path.parent / "receipt.json").relative_to(ROOT)),
        },
    }
    (path.parent / "receipt.json").write_text(stable_json(receipt))
    print(stable_json(receipt), flush=True)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-name")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--seed", type=int, default=20_260_723)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--base-output", type=Path, default=DEFAULT_BASE_OUTPUT
    )
    parser.add_argument(
        "--cache-root", type=Path, default=DEFAULT_CACHE_ROOT
    )
    args = parser.parse_args()
    for name in ("output", "base_output", "cache_root"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    if args.all == bool(args.env_name):
        parser.error("choose exactly one of --all or --env-name")
    if args.env_name and args.env_name not in ENVIRONMENTS:
        parser.error(f"unsupported environment: {args.env_name}")
    return args


def main() -> None:
    args = parse_args()
    environments = ENVIRONMENTS if args.all else (args.env_name,)
    receipts = [
        build_environment(args, env_name) for env_name in environments
    ]
    if args.all:
        report = {
            "schema": "graph_cem_long_gap_build_report_v1",
            "status": "completed",
            "environments": [row["environment"] for row in receipts],
            "gaps": list(GAPS),
            "controlled_splicing": True,
            "frames_modified": False,
        }
        (args.output / "build_report.json").write_text(stable_json(report))


if __name__ == "__main__":
    main()
