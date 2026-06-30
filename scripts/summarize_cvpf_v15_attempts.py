#!/usr/bin/env python3
"""Summarize the failure-aware CVPF-v15 attempt ledger.

This is deliberately separate from the frozen registered analyzer.  The
registered analyzer remains fail-closed when any of the 52 cells is missing.
This script reports descriptive metrics for scientifically complete cells,
including post-failure exact-command completions and sync-only W&B repairs,
without converting them into an official screen result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_cvpf_v15_screen as runner


PRIMARY = "heldout_prior_state_nmse"
CLEAN = "clean_prior_state_nmse"
INTEGRATOR = "initial_encoder_integrator_probe_nmse"
DEFAULT_ROOT = Path("outputs/hacssm_v15_screen_cvpf30")
REQUIRED_ARTIFACTS = ("model.pt", "metrics.json", "eval_rollout.npz", "wandb_run.json")
RUN_ID_PATTERN = re.compile(r"/runs/([a-z0-9]+)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} is boolean")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is non-finite")
    return result


def _run_id(log_text: str, receipt: Mapping[str, Any] | None) -> str | None:
    if receipt is not None and isinstance(receipt.get("run_id"), str):
        return str(receipt["run_id"])
    match = RUN_ID_PATTERN.search(log_text)
    return match.group(1) if match else None


def _last_error(log_text: str) -> str | None:
    for line in reversed(log_text.splitlines()):
        stripped = line.strip()
        if ("Error:" in stripped or stripped.startswith("RuntimeError")
                or stripped.startswith("ValueError")):
            return stripped
    return None


def _cell(root: Path, protocol: Mapping[str, Any], task: str, design: str) -> dict[str, Any]:
    directory = runner.run_directory(root, task, design)
    log_path = ROOT / runner.DEFAULT_LOG_ROOT / (
        f"{runner._slug(task)}-{design}-s{runner.SEED}.log")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    metrics_path = directory / "metrics.json"
    metrics = load_json(metrics_path) if metrics_path.is_file() else None
    receipt_path = directory / "wandb_run.json"
    receipt = load_json(receipt_path) if receipt_path.is_file() else None
    artifacts = {
        name: sha256_file(directory / name)
        for name in REQUIRED_ARTIFACTS if (directory / name).is_file()
    }
    command = protocol["commands"][task][runner.DESIGNS.index(design)]
    epoch_rows = sum(line.startswith("e ") for line in log_text.splitlines())
    complete = metrics is not None and len(artifacts) == len(REQUIRED_ARTIFACTS)
    if complete and receipt and receipt.get("postfailure_sync_repair") is True:
        status = "COMPLETE_SYNC_REPAIRED"
    elif complete:
        status = "COMPLETE"
    elif metrics is not None:
        status = "METRICS_COMPLETE_ARTIFACT_INCOMPLETE"
    else:
        status = "NONFINITE_MODEL_FAILURE"
    row: dict[str, Any] = {
        "task": task,
        "design": design,
        "status": status,
        "command_sha256": runner.json_sha256(command),
        "log": str(log_path),
        "epoch_rows": epoch_rows,
        "run_id": _run_id(log_text, receipt),
        "last_error": _last_error(log_text),
        "artifact_sha256": artifacts,
        "postfailure_sync_repair": bool(
            receipt and receipt.get("postfailure_sync_repair") is True),
        "scientific_metrics_recomputed_by_sync_repair": bool(
            receipt and receipt.get("scientific_metrics_recomputed") is True),
    }
    if metrics is not None:
        row["metrics"] = {
            PRIMARY: finite(metrics.get(PRIMARY), f"{task}/{design}/{PRIMARY}"),
            CLEAN: finite(metrics.get(CLEAN), f"{task}/{design}/{CLEAN}"),
            INTEGRATOR: finite(
                metrics.get(INTEGRATOR), f"{task}/{design}/{INTEGRATOR}"),
            "encoder_covariance_effective_rank": finite(
                metrics.get("encoder_covariance_effective_rank"), task),
            "predictive_loss_convergence_relative_change": finite(
                metrics.get("predictive_loss_convergence_relative_change"), task),
        }
    return row


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [finite(row["metrics"][key], key) for row in rows]
    return sum(values) / len(values)


def _contrast(candidate: Sequence[Mapping[str, Any]], reference: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_task_candidate = {str(row["task"]): finite(row["metrics"][PRIMARY], PRIMARY)
                         for row in candidate}
    by_task_reference = {str(row["task"]): finite(row["metrics"][PRIMARY], PRIMARY)
                         for row in reference}
    tasks = [task for task in runner.TASKS
             if task in by_task_candidate and task in by_task_reference]
    reductions = {
        task: (by_task_reference[task] - by_task_candidate[task])
        / by_task_reference[task]
        for task in tasks
    }
    return {
        "task_count": len(tasks),
        "candidate_mean": sum(by_task_candidate[task] for task in tasks) / len(tasks),
        "reference_mean": sum(by_task_reference[task] for task in tasks) / len(tasks),
        "equal_task_reduction": (
            sum(by_task_reference[task] for task in tasks)
            - sum(by_task_candidate[task] for task in tasks))
        / sum(by_task_reference[task] for task in tasks),
        "wins": sum(by_task_candidate[task] < by_task_reference[task] for task in tasks),
        "task_reductions": reductions,
    }


def summarize(root: Path) -> dict[str, Any]:
    protocol = load_json(root / "screen_protocol.json")
    cells = [
        _cell(root, protocol, task, design)
        for task in runner.TASKS for design in runner.DESIGNS
    ]
    metric_rows = [row for row in cells if "metrics" in row]
    complete_rows = [row for row in cells if row["status"].startswith("COMPLETE")]
    failed_rows = [row for row in cells if row["status"] == "NONFINITE_MODEL_FAILURE"]
    repaired_rows = [row for row in cells if row["postfailure_sync_repair"]]
    by_design = {
        design: [row for row in metric_rows if row["design"] == design]
        for design in runner.DESIGNS
    }
    design_summary: dict[str, Any] = {}
    for design, rows in by_design.items():
        values: dict[str, Any] = {
            "metric_cells": len(rows),
            "missing_tasks": [task for task in runner.TASKS
                              if not any(row["task"] == task for row in rows)],
        }
        if rows:
            values.update({
                "heldout_prior_state_nmse_mean": _mean(rows, PRIMARY),
                "clean_prior_state_nmse_mean": _mean(rows, CLEAN),
                "integrator_nmse_mean": _mean(rows, INTEGRATOR),
                "encoder_effective_rank_mean": _mean(
                    rows, "encoder_covariance_effective_rank"),
                "task_metrics": {
                    str(row["task"]): row["metrics"] for row in rows},
            })
        design_summary[design] = values
    complete_designs = [
        design for design, rows in by_design.items() if len(rows) == len(runner.TASKS)
    ]
    ranking = sorted(
        complete_designs,
        key=lambda design: design_summary[design]["heldout_prior_state_nmse_mean"])
    full = by_design["cvpfv15"]
    contrasts = {
        design: _contrast(full, by_design[design])
        for design in complete_designs if design != "cvpfv15"
    }
    integrator_rows = [
        {"task": row["task"], "metrics": {PRIMARY: row["metrics"][INTEGRATOR]}}
        for row in full
    ]
    full_integrator = _contrast(full, integrator_rows)
    full_metrics = {
        str(row["task"]): load_json(
            runner.run_directory(root, str(row["task"]), "cvpfv15") / "metrics.json")
        for row in full
    }
    mechanism = {}
    representation_passes = 0
    structural_passes = 0
    convergence = {}
    for task, metrics in full_metrics.items():
        action = finite(metrics["cvpf_core_action_gain"], task)
        correction = finite(metrics["cvpf_core_correction_gain"], task)
        suffix = finite(metrics["cvpf_true_action_suffix_advantage"], task)
        pair = finite(metrics["cvpf_action_pair_accuracy"], task)
        rank = finite(metrics["encoder_covariance_effective_rank"], task)
        stream = finite(metrics["cvpf_streaming_max_abs"], task)
        prefix = finite(metrics["cvpf_prefix_closure_max_abs"], task)
        shift = finite(metrics["cvpf_shift_closure_relative"], task)
        exposure = finite(
            metrics["cvpf_core_observation_deployed_to_fit_innovation_rms_ratio"], task)
        representation_passes += int(rank >= 16.0)
        structural_ok = (abs(stream) <= 1e-5 and abs(prefix) <= 1e-5
                         and 0.0 <= shift <= 1.0 + 16.0 * 2.220446049250313e-16
                         and .5 <= exposure <= 2.0)
        structural_passes += int(structural_ok)
        mechanism[task] = {
            "action_gain": action,
            "correction_gain": correction,
            "suffix_advantage": suffix,
            "pair_accuracy": pair,
            "action_gain_passed": action > 0.0,
            "correction_gain_passed": correction > 0.0,
            "suffix_advantage_passed": suffix > 0.0,
            "pair_accuracy_passed": pair > .5,
            "encoder_effective_rank": rank,
            "representation_passed": rank >= 16.0,
            "streaming_max_abs": stream,
            "prefix_closure_max_abs": prefix,
            "shift_closure_relative": shift,
            "innovation_exposure_ratio": exposure,
            "structural_passed": structural_ok,
        }
        convergence[task] = finite(
            metrics["predictive_loss_convergence_relative_change"], task)
    mechanism_counts = {
        key: sum(bool(values[f"{key}_passed"]) for values in mechanism.values())
        for key in ("action_gain", "correction_gain", "suffix_advantage", "pair_accuracy")
    }
    official = load_json(root / "screen_analysis.json")
    audit = load_json(root / "screen_audit.json")
    return {
        "schema_version": 1,
        "scope": "postfailure_descriptive_v15_attempt_ledger",
        "official_status": official.get("status"),
        "official_completed_cells_before_sync_repair": official.get("completed_cells"),
        "official_artifact_integrity_passed": official.get("artifact_integrity_passed"),
        "independent_audit_status": audit.get("status"),
        "official_result": False,
        "expected_cells": len(runner.TASKS) * len(runner.DESIGNS),
        "attempted_cells": len(cells),
        "scientific_metric_cells": len(metric_rows),
        "artifact_complete_cells_after_sync_repair": len(complete_rows),
        "nonfinite_model_failure_cells": len(failed_rows),
        "sync_repaired_cells": len(repaired_rows),
        "automatic_continuation_launch_performed": False,
        "conditional_authorization_status": "CONDITIONAL_NOT_AUTHORIZED",
        "ranking_complete_designs": ranking,
        "design_summary": design_summary,
        "full_contrasts": contrasts,
        "full_vs_legal_integrator": full_integrator,
        "full_mechanism": {
            "required_tasks_per_receipt": 3,
            "passed_task_counts": mechanism_counts,
            "tasks": mechanism,
        },
        "full_representation_passed_tasks": representation_passes,
        "full_structural_passed_tasks": structural_passes,
        "full_convergence_signed_values": convergence,
        "full_convergence_passed": (
            all(value >= 0.0 for value in convergence.values())
            and max(abs(value) for value in convergence.values()) < .05),
        "cells": cells,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--write", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = (args.root if args.root.is_absolute() else ROOT / args.root).resolve()
    result = summarize(root)
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    print(rendered, end="")
    if args.write:
        path = root / "postfailure_analysis.json"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        path.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
