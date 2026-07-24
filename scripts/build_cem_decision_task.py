#!/usr/bin/env python3
"""Build a decision-conditioned action-ranking task from raw PointMaze pairs."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_graph_cem_long_gap import GAPS  # noqa: E402
from scripts.run_cem_raw_ogbench import stable_json  # noqa: E402

OUTPUT = ROOT / "outputs/cem_decision_memory_v1"
SOURCE = ROOT / "outputs/graph_cem_long_gap_v1"
BASE = ROOT / "outputs/cem_raw_ogbench"
ENV = "pointmaze-large-navigate-v0"


def source_audit() -> dict[str, Any]:
    tree = ast.parse(Path(__file__).read_text())
    loaded = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    forbidden = {
        "cue_labels", "cue_positions", "cue_window", "reward", "goal_state",
    }
    violations = sorted(forbidden & loaded)
    return {
        "passed": not violations,
        "forbidden_loaded_names": violations,
        "visual_cue_overlay": False,
        "manual_event_label": False,
        "known_event_time_model_input": False,
        "construction": (
            "existing unmodified-frame controlled suffix-collision recipe; "
            "decision candidates added without changing histories"
        ),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    source = SOURCE / "build" / ENV / "pairs.npz"
    source_receipt = SOURCE / "build" / ENV / "receipt.json"
    feature = BASE / "features" / ENV / "features.npz"
    if not source.is_file() or not feature.is_file():
        raise FileNotFoundError("required fixed long-gap artifacts are missing")
    output = args.output / "build" / ENV
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output / "pairs.npz")
    with np.load(source, allow_pickle=False) as data:
        split_counts = {
            split: int(len(data[f"{split}_sources"]))
            for split in ("train", "validation", "test")
        }
    with np.load(feature, allow_pickle=False) as data:
        actions = np.asarray(data["actions"])
        latent_shape = list(data["latents"].shape)
    if actions.shape[1] < 21:
        raise ValueError("source actions do not support four-step candidates")
    audit = source_audit()
    if not audit["passed"]:
        raise RuntimeError(f"decision source audit failed: {audit}")
    receipt = {
        "schema": "cem_decision_task_build_v1",
        "status": "completed",
        "environment": ENV,
        "gaps": [value for value in GAPS if value >= 32],
        "split_pair_counts": split_counts,
        "feature_shape": latent_shape,
        "candidate_actions": (
            "branch-own native source actions, paired-branch source actions, "
            "shared donor actions, deterministic unrelated source actions"
        ),
        "goal_query": "branch-specific realized future DINO latent (evaluator task query)",
        "recent_suffix": "exact paired six-frame latent/action suffix",
        "native_renderings_modified": False,
        "native_state_action_source": True,
        "native_chronology": False,
        "controlled_splicing_disclosed": True,
        "executed_controller_supported": False,
        "executed_controller_reason": (
            "admitted PointMaze controller supports standard task goal IDs, "
            "not the controlled branch-future action target"
        ),
        "source_recipe_receipt": str(source_receipt.relative_to(ROOT)),
        "source_contract": audit,
        "artifacts": {
            "pairs": str((output / "pairs.npz").relative_to(ROOT)),
            "receipt": str((output / "receipt.json").relative_to(ROOT)),
        },
    }
    (output / "receipt.json").write_text(stable_json(receipt))
    print(stable_json(receipt))
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not args.output.is_absolute():
        args.output = ROOT / args.output
    return args


if __name__ == "__main__":
    build(parse_args())
