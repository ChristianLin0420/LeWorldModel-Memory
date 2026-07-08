#!/usr/bin/env python3
"""Pre-lock PointMaze controller-horizon selection on training states only."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.official_tasks.dinowm_pointmaze import (  # noqa: E402
    GOAL_WAYPOINTS,
    CurrentMujocoPointMaze,
    execute_released_waypoint,
)
from scripts.run_dinowm_pointmaze_wave3 import NativePointMazeData  # noqa: E402


CONFIG = ROOT / "configs/dinowm_pointmaze_wave3.yaml"


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    receipt = ROOT / cfg["external_use"]["prelock_controller_development"][
        "receipt"]
    if receipt.exists():
        raise FileExistsError(f"refusing to overwrite {receipt}")
    if CONFIG.with_suffix(".lock.json").exists():
        raise RuntimeError("controller development is forbidden after locking")
    dataset = NativePointMazeData(cfg)
    selections = dataset.selections()[:8]
    if any(value.split != "train" for value in selections):
        raise RuntimeError("controller development opened validation data")
    states = np.stack([dataset.read(value)["state"][18]
                       for value in selections])
    vendor = ROOT / cfg["source"]["dino_wm"]["repo_path"]
    simulator = CurrentMujocoPointMaze(vendor)
    records = []
    chosen = None
    development = cfg["external_use"]["prelock_controller_development"]
    for horizon in development["candidate_horizons"]:
        success = np.empty((len(states), 4), dtype=np.int8)
        steps = np.empty((len(states), 4), dtype=np.int32)
        for row, state in enumerate(states):
            for label, target in enumerate(GOAL_WAYPOINTS):
                result = execute_released_waypoint(
                    simulator, vendor, initial_state=state, target=target,
                    horizon=int(horizon), controller_seed=8_300_000 + row * 4 + label,
                    success_radius=float(cfg["external_use"]["success_radius"]))
                success[row, label] = int(result["success"])
                steps[row, label] = int(result["steps"])
        overall = float(success.mean())
        per_goal = success.mean(axis=0).astype(float).tolist()
        passed = overall >= 0.90 and min(per_goal) >= 0.875
        records.append({"horizon": int(horizon), "overall": overall,
                        "per_goal": per_goal, "passed": bool(passed),
                        "mean_steps": float(steps.mean()),
                        "max_steps": int(steps.max())})
        if chosen is None and passed:
            chosen = int(horizon)
    payload = {
        "schema": "dinowm_pointmaze_controller_development_v1",
        "status": "passed" if chosen is not None else "failed",
        "training_selections": [
            {"episode_index": value.episode_index,
             "local_start": value.local_start} for value in selections],
        "validation_opened": False, "goal_cue_metric_computed": False,
        "carrier_metric_computed": False, "candidates": records,
        "chosen_horizon": chosen,
        "selection_rule": development["selection_rule"],
        "current_mujoco_version": simulator.mujoco.__version__,
        "released_xml_sha256": simulator.xml_sha256,
    }
    receipt.parent.mkdir(parents=True, exist_ok=False)
    temporary = receipt.with_name(f".{receipt.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, receipt)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if chosen is None:
        raise SystemExit("no controller horizon passed on training states")


if __name__ == "__main__":
    main()
