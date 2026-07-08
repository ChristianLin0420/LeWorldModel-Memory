#!/usr/bin/env python3
"""Aggregate V2 only after every cell uses the same sealed controller."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.aggregate_paper_a_delayed_goal_use import (
    _task_summary,
    _validate_metric_record,
)
from scripts.paper_a_delayed_goal_spec import SOURCE_IDS, source_slug
from scripts.paper_a_delayed_goal_v2_spec import (
    DEFAULT_SPEC,
    SEEDS,
    TASKS,
    evaluation_directory,
    load_controller_lock,
    load_locked_spec,
    resolve_path,
    sha256_file,
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def _read_cell(path: Path, spec: Mapping[str, Any], task: str, seed: int,
               controller_record: Mapping[str, str],
               controller_lock: Mapping[str, Any]) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read V2 evaluation {path}: {error}") from error
    expected = {
        "study": spec["study"], "task": task, "checkpoint_seed": seed,
        "spec": spec["_spec_record"], "controller_lock": controller_record,
        "selected_candidate_id": controller_lock["selected_candidate_id"],
        "selected_protocol": controller_lock["selected_protocol"],
        "v1_failure_provenance": spec["v1"]["provenance_manifest"],
        "v1_repairs_reused": True, "repair_retraining_performed": False,
    }
    failed = [key for key, value in expected.items()
              if result.get(key) != value]
    if failed:
        raise ValueError(f"V2 cell identity mismatch {path}: {failed}")
    labels = np.asarray(result.get("labels"), dtype=np.int64)
    if labels.shape != (240,) or not np.isin(labels, np.arange(4)).all():
        raise ValueError(f"invalid V2 labels in {path}")
    expected_slugs = {source_slug(spec, source) for source in SOURCE_IDS}
    if set(result.get("sources", {})) != expected_slugs \
            or result.get("consumer", {}).get("source_order") != list(SOURCE_IDS) \
            or result.get("consumer", {}).get(
                "shared_across_all_sources") is not True:
        raise ValueError(f"V2 shared-consumer/source grid mismatch in {path}")
    for source in SOURCE_IDS:
        slug = source_slug(spec, source)
        record = result["sources"][slug]
        if record.get("id") != source:
            raise ValueError(f"V2 source identity mismatch in {path}")
        _validate_metric_record(
            record.get("metrics", {}), labels, f"{path}/{source}")
    controls = result.get("controls", {})
    shuffled = controls.get("label_shuffle", {})
    if set(shuffled) != expected_slugs:
        raise ValueError(f"V2 shuffle grid mismatch in {path}")
    for source in SOURCE_IDS:
        slug = source_slug(spec, source)
        _validate_metric_record(
            shuffled[slug].get("metrics", {}), labels,
            f"{path}/shuffle/{source}")
    _validate_metric_record(
        controls.get("action_time", {}), labels, f"{path}/action-time")
    oracle = result.get("label_oracle", {})
    _validate_metric_record(oracle, labels, f"{path}/oracle")
    if not np.array_equal(np.asarray(oracle.get("prediction")), labels):
        raise ValueError(f"V2 oracle prediction mismatch in {path}")
    limit = float(spec["controls"]["shortcut_accuracy_max"])
    max_shuffle = max(
        shuffled[slug]["metrics"]["goal_decision_accuracy"]
        for slug in expected_slugs)
    expected_validity = {
        "label_oracle_success_pass": (
            oracle["executed_success_rate"] >=
            float(spec["executed_choice"]["validation_oracle_success_min"])),
        "label_shuffle_accuracy_pass": max_shuffle <= limit,
        "action_time_accuracy_pass": (
            controls["action_time"]["goal_decision_accuracy"] <= limit),
        "max_label_shuffle_accuracy": max_shuffle,
        "shortcut_accuracy_limit": limit,
    }
    expected_validity["valid_for_use_claim"] = all(
        expected_validity[key] for key in (
            "label_oracle_success_pass", "label_shuffle_accuracy_pass",
            "action_time_accuracy_pass"))
    if result.get("validity") != expected_validity:
        raise ValueError(f"V2 validity receipt mismatch in {path}")
    return result


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.execute:
        raise SystemExit("refusing V2 aggregation without explicit --execute")
    spec = load_locked_spec(args.spec)
    controller_lock, controller_record = load_controller_lock(spec)
    output = resolve_path(spec["output"]["summary"])
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    tasks: dict[str, Any] = {}
    source_evaluations: list[dict[str, str]] = []
    for task in TASKS:
        cells = []
        for seed in SEEDS:
            path = evaluation_directory(spec, task, seed) / "metrics.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing expected V2 evaluation {path}")
            cells.append(_read_cell(
                path, spec, task, seed, controller_record, controller_lock))
            source_evaluations.append({
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
            })
        tasks[task] = _task_summary(spec, task, cells)
    all_gates = all(
        task["controls"]["all_cell_validity_gates_pass"]
        for task in tasks.values())
    summary = {
        "schema_version": 1,
        "study": spec["study"],
        "spec": spec["_spec_record"],
        "status": "valid" if all_gates else "invalid_controls_or_oracle",
        "controller_lock": controller_record,
        "selected_candidate_id": controller_lock["selected_candidate_id"],
        "selected_protocol": controller_lock["selected_protocol"],
        "v1_failure_provenance": spec["v1"]["provenance_manifest"],
        "v1_repairs_reused": True,
        "repair_retraining_performed": False,
        "inference": spec["endpoints"]["bootstrap"],
        "tasks": tasks,
        "source_evaluations": source_evaluations,
        "claim_boundary": spec["claim_boundary"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            "w", dir=output.parent, prefix=f".{output.name}.",
            suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    try:
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(f"[delayed-goal-v2] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
