#!/usr/bin/env python3
"""Release held-out PushT labels after gates and aggregate downstream use."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import yaml
from sklearn.metrics import balanced_accuracy_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.official_tasks.artifacts import sha256_file  # noqa: E402
from lewm.official_tasks.pusht_downstream import stable_json  # noqa: E402
from lewm.official_tasks.pusht_pipeline import (  # noqa: E402
    aligned_pusht_latents,
    load_pusht_base_cache,
    load_pusht_task_cache,
)
from lewm.official_tasks.pusht_spec import load_locked_pusht_spec  # noqa: E402


SPEC_PATH = ROOT / "configs/paper_a_pusht_downstream_use_v1.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def _crossed_ci(values: np.ndarray, *, replicates: int,
                seed: int) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("crossed bootstrap input must be seed x episode")
    rng = np.random.default_rng(seed)
    out = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        seeds = rng.integers(0, values.shape[0], values.shape[0])
        episodes = rng.integers(0, values.shape[1], values.shape[1])
        out[index] = values[np.ix_(seeds, episodes)].mean()
    return [float(value) for value in np.quantile(out, (0.025, 0.975))]


def _feature_path(root: Path, task: str, arm: str, seed: int) -> Path:
    return root / "features" / task / arm / f"seed-{seed}.npz"


def _prediction_path(root: Path, task: str, arm: str, seed: int) -> Path:
    return root / "predictions" / task / arm / f"seed-{seed}.npz"


def main() -> None:
    args = parse_args()
    if not args.execute:
        raise RuntimeError("formal evaluation writes artifacts; pass --execute")
    spec = yaml.safe_load(SPEC_PATH.read_text())
    protocol_sha = sha256_file(SPEC_PATH)
    expected = (SPEC_PATH.with_suffix(".sha256").read_text().split()[0])
    if protocol_sha != expected:
        raise RuntimeError("downstream protocol no longer matches its lock")
    root = ROOT / spec["artifacts"]["root"]
    summary_path = ROOT / spec["artifacts"]["summary"]
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite {summary_path}")
    receipt_path = ROOT / spec["artifacts"]["protocol_receipt"]
    gates_path = ROOT / spec["artifacts"]["gates"]
    receipt = json.loads(receipt_path.read_text())
    gates = json.loads(gates_path.read_text())
    if receipt["protocol"]["sha256"] != protocol_sha:
        raise RuntimeError("protocol receipt belongs to a different lock")
    if gates.get("physical_development_pending", True):
        raise RuntimeError("physical development gate is incomplete")

    parent = load_locked_pusht_spec()
    base_test, _ = load_pusht_base_cache(parent, "validation")
    arms = list(spec["source_audit"]["carriers"]["arms"])
    seeds = [int(value) for value in spec["source_audit"]["carriers"]["seeds"]]
    reps = int(spec["metrics"]["bootstrap"]["replicates"])
    bootstrap_seed = int(spec["metrics"]["bootstrap"]["seed"])
    all_results: dict[str, dict] = {}
    artifact_records: list[dict] = []

    for task_index, task_record in enumerate(spec["tasks"]):
        task = task_record["key"]
        classes = int(task_record["classes"])
        gate = gates["tasks"][task]
        released = bool(gate["semantic_pass"] and gate["physical_oracle_pass"])
        if not released:
            all_results[task] = {
                "status": "stopped_before_heldout_labels",
                "semantic_pass": gate["semantic_pass"],
                "physical_oracle_pass": gate["physical_oracle_pass"],
            }
            continue

        # This is the first point in the workflow where held-out semantic labels
        # are loaded and predictions are scored.
        task_test, _ = load_pusht_task_cache(parent, task, "validation")
        z_test = aligned_pusht_latents(base_test, task_test, 1, 3)
        raw_test = z_test[:, 16:19].reshape(len(z_test), -1).astype(np.float32)
        labels_full = task_test["labels"].astype(np.int64)
        physical_path = (root / "simulator" / "test" / f"{task}.npz")
        physical = np.load(physical_path)
        if not np.array_equal(
                physical["episode_ids"],
                base_test["episode_index"][physical["source_rows"]]):
            raise RuntimeError("physical and semantic held-out rows differ")
        rows = physical["source_rows"].astype(np.int64)
        labels = labels_full[rows]
        success_matrix = physical["success_matrix"]
        regret_matrix = physical["regret_matrix"]
        n = len(rows)

        selection: dict[str, list[np.ndarray]] = {arm: [] for arm in arms}
        executed: dict[str, list[np.ndarray]] = {arm: [] for arm in arms}
        regret: dict[str, list[np.ndarray]] = {arm: [] for arm in arms}
        balanced: dict[str, list[float]] = {arm: [] for arm in arms}
        for seed in seeds:
            model_path = root / "consumers" / task / f"seed-{seed}.joblib"
            model = joblib.load(model_path)
            for arm in arms:
                feature_path = _feature_path(root, task, arm, seed)
                feature = np.load(feature_path)
                if not np.array_equal(feature["episode_test"],
                                      base_test["episode_index"]):
                    raise RuntimeError(f"feature episode ids differ: {feature_path}")
                x = np.concatenate(
                    (raw_test, feature["prior_test"]), axis=1)
                prediction_full = model.predict(x).astype(np.int64)
                prediction = prediction_full[rows]
                episode_axis = np.arange(n)
                success = success_matrix[episode_axis, prediction, labels]
                pose_regret = regret_matrix[episode_axis, prediction, labels]
                selection[arm].append((prediction == labels).astype(np.float64))
                executed[arm].append(success.astype(np.float64))
                regret[arm].append(pose_regret.astype(np.float64))
                balanced[arm].append(float(balanced_accuracy_score(
                    labels, prediction)))
                output = _prediction_path(root, task, arm, seed)
                output.parent.mkdir(parents=True, exist_ok=True)
                if output.exists():
                    raise FileExistsError(f"refusing to overwrite {output}")
                np.savez_compressed(
                    output,
                    source_rows=rows,
                    episode_ids=physical["episode_ids"],
                    labels=labels,
                    predictions=prediction,
                    executed_success=success,
                    pose_regret=pose_regret,
                )
                artifact_records.append({
                    "path": str(output.relative_to(ROOT)),
                    "sha256": sha256_file(output),
                })

        random_rng = np.random.default_rng(
            int(spec["baselines_and_controls"]["random_goal"]["seed"])
            + task_index)
        random_goal = random_rng.integers(0, classes, size=n)
        random_success = success_matrix[
            np.arange(n), random_goal, labels].astype(np.float64)
        oracle_success = success_matrix[
            np.arange(n), labels, labels].astype(np.float64)

        arms_result: dict[str, dict] = {}
        for arm_index, arm in enumerate(arms):
            select_values = np.stack(selection[arm])
            success_values = np.stack(executed[arm])
            regret_values = np.stack(regret[arm])
            result = {
                "goal_selection_accuracy": float(select_values.mean()),
                "goal_selection_balanced_accuracy": float(np.mean(balanced[arm])),
                "executed_success": float(success_values.mean()),
                "mean_block_pose_regret": float(regret_values.mean()),
                "per_seed": [
                    {
                        "seed": seed,
                        "goal_selection_accuracy": float(select_values[index].mean()),
                        "goal_selection_balanced_accuracy": balanced[arm][index],
                        "executed_success": float(success_values[index].mean()),
                        "mean_block_pose_regret": float(regret_values[index].mean()),
                    }
                    for index, seed in enumerate(seeds)
                ],
            }
            if arm != "none":
                base_selection = np.stack(selection["none"])
                base_success = np.stack(executed["none"])
                base_regret = np.stack(regret["none"])
                select_diff = select_values - base_selection
                success_diff = success_values - base_success
                regret_reduction = base_regret - regret_values
                result["contrast_vs_none"] = {
                    "goal_selection_accuracy": {
                        "estimate": float(select_diff.mean()),
                        "ci95": _crossed_ci(
                            select_diff, replicates=reps,
                            seed=bootstrap_seed + 100 * task_index + arm_index),
                    },
                    "executed_success": {
                        "estimate": float(success_diff.mean()),
                        "ci95": _crossed_ci(
                            success_diff, replicates=reps,
                            seed=bootstrap_seed + 1000 + 100 * task_index + arm_index),
                    },
                    "block_pose_regret_reduction": {
                        "estimate": float(regret_reduction.mean()),
                        "ci95": _crossed_ci(
                            regret_reduction, replicates=reps,
                            seed=bootstrap_seed + 2000 + 100 * task_index + arm_index),
                    },
                }
            arms_result[arm] = result

        all_results[task] = {
            "status": "heldout_complete",
            "classes": classes,
            "eligible_episodes": n,
            "eligible_fraction": float(n / len(base_test["episode_index"])),
            "oracle_executed_success": float(oracle_success.mean()),
            "random_goal_executed_success": float(random_success.mean()),
            "random_goal_expected_chance": 1.0 / classes,
            "arms": arms_result,
            "physical_artifact": {
                "path": str(physical_path.relative_to(ROOT)),
                "sha256": sha256_file(physical_path),
            },
        }
        gates["tasks"][task]["formal_test_released"] = True

    summary = {
        "schema": "paper_a_pusht_downstream_summary_v1",
        "protocol_sha256": protocol_sha,
        "estimand": "belief-conditioned PushT goal selection under one locked native-physics controller",
        "device": "cuda:1 for carrier feature extraction; pinned CPU simulator for execution",
        "gates": gates["tasks"],
        "tasks": all_results,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(stable_json(summary))
    gates_path.write_text(stable_json(gates))

    markdown = [
        "# PushT downstream-use extension (locked v1)", "",
        f"Protocol SHA-256: `{protocol_sha}`", "",
    ]
    for task, result in all_results.items():
        markdown.extend([f"## {task}", ""])
        if result["status"] != "heldout_complete":
            markdown.extend([f"Stopped: `{result}`", ""])
            continue
        markdown.extend([
            f"Eligible episodes: {result['eligible_episodes']}; "
            f"oracle success {result['oracle_executed_success']:.3f}; "
            f"random-goal success {result['random_goal_executed_success']:.3f}.",
            "",
            "| Carrier | selection | executed success | pose regret | Δ success vs none [95% CI] |",
            "|---|---:|---:|---:|---:|",
        ])
        for arm, record in result["arms"].items():
            if arm == "none":
                contrast = "--"
            else:
                value = record["contrast_vs_none"]["executed_success"]
                contrast = (f"{value['estimate']:+.3f} "
                            f"[{value['ci95'][0]:+.3f}, {value['ci95'][1]:+.3f}]")
            markdown.append(
                f"| {arm} | {record['goal_selection_accuracy']:.3f} | "
                f"{record['executed_success']:.3f} | "
                f"{record['mean_block_pose_regret']:.2f} | {contrast} |")
        markdown.append("")
    markdown_path = ROOT / spec["artifacts"]["summary_markdown"]
    markdown_path.write_text("\n".join(markdown) + "\n")

    producer_paths = [
        SPEC_PATH,
        ROOT / "lewm/official_tasks/pusht_downstream.py",
        ROOT / "scripts/prepare_paper_a_pusht_downstream_use.py",
        ROOT / "scripts/simulate_paper_a_pusht_downstream_use.py",
        ROOT / "scripts/evaluate_paper_a_pusht_downstream_use.py",
        receipt_path,
        gates_path,
        summary_path,
        markdown_path,
    ]
    provenance = {
        "schema": "paper_a_pusht_downstream_provenance_v1",
        "protocol_sha256": protocol_sha,
        "stable_worldmodel_commit": spec["upstream_simulator"]["commit"],
        "workspace_git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True).stdout.strip(),
        "artifacts": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
            for path in producer_paths
        ] + artifact_records,
    }
    provenance_path = ROOT / spec["artifacts"]["provenance"]
    provenance_path.write_text(stable_json(provenance))
    print(stable_json(summary))


if __name__ == "__main__":
    main()
