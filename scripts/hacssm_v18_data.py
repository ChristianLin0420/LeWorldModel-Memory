#!/usr/bin/env python3
"""Unopened DMC trajectory cohort for the LeWM+V8 confirmation study.

The storage schema, IID bounded-action process, deterministic corruption views,
and evaluation-only native task-state targets are exactly the V11 data contract.
Only the task set and data/corruption seeds change.  Keeping this adapter in a
separate module leaves the frozen V11--V17 task registries untouched.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.hacssm_v11_data as v11


TASKS = (
    "acrobot.swingup",
    "manipulator.bring_ball",
    "quadruped.run",
    "stacker.stack_4",
    "swimmer.swimmer15",
)
DEFAULT_ROOT = "outputs/lewm_v8_v18_data"
DEFAULT_TRAIN_EPISODES = 1_200
DEFAULT_VAL_EPISODES = 240
DEFAULT_LENGTH = 48
DEFAULT_IMG_SIZE = 64
DEFAULT_TRAIN_SEED = 270_701
DEFAULT_VAL_SEED = 270_702
DEFAULT_SMOOTH_RHO = 0.0
DEFAULT_CORRUPTION_SEED = 270_711

ACTION_PROCESS = v11.ACTION_PROCESS
VIEWS = v11.VIEWS
cache_name = v11.cache_name
sha256_file = v11.sha256_file


@contextlib.contextmanager
def _v18_task_registry():
    """Temporarily expose the frozen V18 whitelist to V11 collection helpers.

    V11 validates collection and cache writing through its module-level task
    registry.  Keeping the replacement scoped to one collection call avoids
    changing the V11--V17 registry merely by importing this adapter.
    """
    previous = v11.TASKS
    v11.TASKS = TASKS
    try:
        yield
    finally:
        v11.TASKS = previous


def _collect_one(*args, **kwargs) -> dict[str, Any]:
    with _v18_task_registry():
        return v11._collect_one(*args, **kwargs)


def load_cache(path: str | Path, *, verify: bool = True,
               return_values: bool = False):
    """Load a V11-schema cache and require membership in the V18 cohort."""
    result = v11.load_cache(
        path, verify=verify, return_values=return_values)
    metadata = result[1] if return_values else result
    if metadata.env_id not in TASKS:
        raise ValueError(
            f"V18 cache task {metadata.env_id!r} is not in the frozen cohort")
    return result


class V18TrajectoryDataset(v11.V11TrajectoryDataset):
    """V11 deterministic corruption views restricted to frozen V18 tasks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.metadata.env_id not in TASKS:
            raise ValueError(
                f"V18 dataset task {self.metadata.env_id!r} is not in the frozen cohort")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--env", action="append", choices=TASKS)
    parser.add_argument("--split", choices=("train", "val", "both"), default="both")
    parser.add_argument("--train-episodes", type=int, default=DEFAULT_TRAIN_EPISODES)
    parser.add_argument("--val-episodes", type=int, default=DEFAULT_VAL_EPISODES)
    parser.add_argument("--length", type=int, default=DEFAULT_LENGTH)
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--train-seed", type=int, default=DEFAULT_TRAIN_SEED)
    parser.add_argument("--val-seed", type=int, default=DEFAULT_VAL_SEED)
    parser.add_argument("--smooth-rho", type=float, default=DEFAULT_SMOOTH_RHO)
    parser.add_argument(
        "--no-manifest", action="store_true",
        help="collect a disjoint task shard without writing the cohort manifest")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if not math.isfinite(args.smooth_rho) or args.smooth_rho != 0.0:
        raise ValueError("V18 collection requires IID actions (--smooth-rho 0.0)")
    environments = TASKS if args.all else tuple(args.env or ())
    if not environments:
        raise ValueError("select --all or at least one --env")
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    splits = ("train", "val") if args.split == "both" else (args.split,)
    records = []
    for env_id in environments:
        for split in splits:
            episodes = args.train_episodes if split == "train" else args.val_episodes
            seed = args.train_seed if split == "train" else args.val_seed
            print(
                f"collect/validate V18 {env_id} {split}: n={episodes} "
                f"L={args.length}", flush=True)
            records.append(_collect_one(
                root, env_id, split, episodes, args.length, args.img_size,
                seed, args.smooth_rho))
    if not args.no_manifest:
        protocol = {
            "study": "lewm-v8-v18-confirmation",
            "scope": "prospectively_frozen_unopened_task_confirmation",
            "tasks": list(environments),
            "splits": list(splits),
            "train_episodes": args.train_episodes,
            "val_episodes": args.val_episodes,
            "length": args.length,
            "img_size": args.img_size,
            "train_seed": args.train_seed,
            "val_seed": args.val_seed,
            "smooth_rho": 0.0,
            "action_process": ACTION_PROCESS,
            "primary_evaluation_target": "flattened_native_task_observation",
            "secondary_evaluation_target": "raw_physics_state",
            "evaluation_targets_used_for_training": False,
            "cache_role": "clean_only_corruptions_are_deterministic_dataset_views",
            "corruption_seed": DEFAULT_CORRUPTION_SEED,
        }
        v11._write_manifest(root, records, protocol)
    print(json.dumps({
        "root": str(root), "tasks": list(environments),
        "manifest_written": not args.no_manifest, "artifacts": records,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
