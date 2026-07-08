#!/usr/bin/env python3
"""Fail-closed numerical and provenance audit for PushT downstream use."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.official_tasks.artifacts import sha256_file  # noqa: E402
from lewm.official_tasks.pusht_downstream import stable_json  # noqa: E402


OUTPUT = ROOT / "outputs/pusht_downstream_use_v1"
TASKS = ("transient-visual-token-recall",
         "multi-item-visual-binding-recall")
ARMS = ("none", "gru", "lstm", "ssm", "fixed_trust")
SEEDS = range(5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def _close(actual: float, expected: float, name: str) -> None:
    if not np.isclose(actual, expected, atol=1e-12, rtol=0):
        raise RuntimeError(f"{name} differs: {actual} != {expected}")


def main() -> None:
    args = parse_args()
    if not args.execute:
        raise RuntimeError("audit writes receipts; pass --execute")
    audit_path = OUTPUT / "final_audit.json"
    environment_path = OUTPUT / "simulator_environment.json"
    if audit_path.exists() or environment_path.exists():
        raise FileExistsError("final PushT downstream audit already exists")

    config = ROOT / "configs/paper_a_pusht_downstream_use_v1.yaml"
    expected_sha, expected_name = config.with_suffix(".sha256").read_text().split()
    if expected_name != config.name or sha256_file(config) != expected_sha:
        raise RuntimeError("formal configuration lock fails")
    summary = json.loads((OUTPUT / "summary.json").read_text())
    gates = json.loads((OUTPUT / "gates.json").read_text())
    receipt = json.loads((OUTPUT / "protocol_receipt.json").read_text())
    provenance = json.loads((OUTPUT / "provenance.json").read_text())
    if not all(gates["tasks"][task]["formal_test_released"] for task in TASKS):
        raise RuntimeError("a held-out task was not gate-released")
    if expected_sha != summary["protocol_sha256"] \
            or expected_sha != receipt["protocol"]["sha256"] \
            or expected_sha != provenance["protocol_sha256"]:
        raise RuntimeError("protocol hash is inconsistent across receipts")

    inventory: list[dict] = []
    numerical: dict[str, dict] = {}
    for task in TASKS:
        physical_path = OUTPUT / "simulator/test" / f"{task}.npz"
        physical = np.load(physical_path)
        rows = physical["source_rows"]
        if len(np.unique(physical["episode_ids"])) != len(rows):
            raise RuntimeError(f"physical episodes repeat for {task}")
        task_result = summary["tasks"][task]
        if int(task_result["eligible_episodes"]) != len(rows):
            raise RuntimeError(f"eligible count differs for {task}")
        task_audit: dict[str, dict] = {}
        for arm in ARMS:
            success_rows, selection_rows, regret_rows = [], [], []
            for seed in SEEDS:
                feature_path = (OUTPUT / "features" / task / arm
                                / f"seed-{seed}.npz")
                prediction_path = (OUTPUT / "predictions" / task / arm
                                   / f"seed-{seed}.npz")
                feature = np.load(feature_path)
                prediction = np.load(prediction_path)
                if feature["prior_train"].shape != (1200, 192) \
                        or feature["prior_test"].shape != (480, 192):
                    raise RuntimeError(f"feature shape differs: {feature_path}")
                if not np.array_equal(prediction["source_rows"], rows) \
                        or not np.array_equal(
                            prediction["episode_ids"], physical["episode_ids"]):
                    raise RuntimeError(f"prediction rows differ: {prediction_path}")
                index = np.arange(len(rows))
                recomputed_success = physical["success_matrix"][
                    index, prediction["predictions"], prediction["labels"]]
                recomputed_regret = physical["regret_matrix"][
                    index, prediction["predictions"], prediction["labels"]]
                if not np.array_equal(
                        recomputed_success, prediction["executed_success"]):
                    raise RuntimeError(f"success indexing differs: {prediction_path}")
                if not np.allclose(
                        recomputed_regret, prediction["pose_regret"], atol=1e-5):
                    raise RuntimeError(f"regret indexing differs: {prediction_path}")
                success_rows.append(prediction["executed_success"].astype(float))
                selection_rows.append(
                    (prediction["predictions"] == prediction["labels"]).astype(float))
                regret_rows.append(prediction["pose_regret"].astype(float))
                for path in (feature_path, prediction_path):
                    inventory.append({
                        "path": str(path.relative_to(ROOT)),
                        "sha256": sha256_file(path),
                    })
            success = float(np.mean(success_rows))
            selection = float(np.mean(selection_rows))
            regret = float(np.mean(regret_rows))
            published = task_result["arms"][arm]
            _close(success, published["executed_success"],
                   f"{task}/{arm}/success")
            _close(selection, published["goal_selection_accuracy"],
                   f"{task}/{arm}/selection")
            _close(regret, published["mean_block_pose_regret"],
                   f"{task}/{arm}/regret")
            task_audit[arm] = {
                "executed_success": success,
                "selection_accuracy": selection,
                "mean_pose_regret": regret,
            }
        numerical[task] = task_audit
        inventory.append({
            "path": str(physical_path.relative_to(ROOT)),
            "sha256": sha256_file(physical_path),
        })

    consumer_paths = sorted((OUTPUT / "consumers").rglob("*.joblib"))
    if len(consumer_paths) != 10:
        raise RuntimeError("expected ten common arm-blind consumers")
    inventory.extend({
        "path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)
    } for path in consumer_paths)
    for record in provenance["artifacts"]:
        path = ROOT / record["path"]
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"provenance artifact changed: {path}")

    vendor = OUTPUT / "vendor/stable-worldmodel"
    vendor_commit = subprocess.run(
        ["git", "-C", str(vendor), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True).stdout.strip()
    vendor_status = subprocess.run(
        ["git", "-C", str(vendor), "status", "--porcelain"], check=True,
        capture_output=True, text=True).stdout.strip()
    if vendor_commit != provenance["stable_worldmodel_commit"] or vendor_status:
        raise RuntimeError("pinned simulator checkout is changed or dirty")
    sim_python = OUTPUT / "simulator_venv/bin/python"
    freeze = subprocess.run(
        [str(sim_python), "-m", "pip", "freeze", "--local"], check=True,
        capture_output=True, text=True).stdout.strip().splitlines()
    environment = {
        "schema": "paper_a_pusht_simulator_environment_v1",
        "python": subprocess.run(
            [str(sim_python), "--version"], check=True,
            capture_output=True, text=True).stdout.strip(),
        "packages": sorted(freeze),
        "stable_worldmodel_commit": vendor_commit,
        "stable_worldmodel_dirty": False,
    }
    environment_path.write_text(stable_json(environment))

    core_paths = [
        config, config.with_suffix(".sha256"),
        ROOT / "lewm/official_tasks/pusht_downstream.py",
        ROOT / "scripts/prepare_paper_a_pusht_downstream_use.py",
        ROOT / "scripts/simulate_paper_a_pusht_downstream_use.py",
        ROOT / "scripts/evaluate_paper_a_pusht_downstream_use.py",
        ROOT / "scripts/audit_paper_a_pusht_downstream_use.py",
        ROOT / "tests/test_paper_a_pusht_downstream_use.py",
        OUTPUT / "protocol_receipt.json", OUTPUT / "gates.json",
        OUTPUT / "state_reset_receipt.json",
        OUTPUT / "simulator/test_receipt.json",
        OUTPUT / "summary.json", OUTPUT / "summary.md",
        OUTPUT / "provenance.json", environment_path,
    ]
    inventory.extend({
        "path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)
    } for path in core_paths)
    audit = {
        "schema": "paper_a_pusht_downstream_final_audit_v1",
        "status": "complete",
        "protocol_sha256": expected_sha,
        "physical_cuda_device": receipt["physical_cuda_device"],
        "stable_worldmodel_commit": vendor_commit,
        "formal_tasks_released": list(TASKS),
        "numerical_recomputation": numerical,
        "artifact_count": len(inventory),
        "artifacts": sorted(inventory, key=lambda value: value["path"]),
    }
    audit_path.write_text(stable_json(audit))
    print(stable_json({
        "status": "complete", "protocol_sha256": expected_sha,
        "artifact_count": len(inventory),
        "audit": str(audit_path.relative_to(ROOT)),
    }))


if __name__ == "__main__":
    main()
