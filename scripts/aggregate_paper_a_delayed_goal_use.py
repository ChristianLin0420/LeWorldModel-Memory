#!/usr/bin/env python3
"""Aggregate the locked delayed-goal use study with crossed paired CIs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.paper_a_delayed_goal_spec import (
    DEFAULT_SPEC,
    SEEDS,
    SOURCE_IDS,
    TASKS,
    evaluation_directory,
    load_locked_spec,
    resolve_path,
    sha256_file,
    source_slug,
)


ARRAY_ENDPOINTS = {
    "goal_decision_accuracy": "correct",
    "executed_success_rate": "success",
    "mean_executed_return": "executed_return",
    "mean_regret_to_label_oracle": "regret_to_label_oracle",
}
COMPARISONS = (
    ("gru_vs_none", "gru", "none"),
    ("ssm_vs_none", "ssm", "none"),
    ("fixed_trust_vs_none", "fixed_trust", "none"),
    ("gru_cue_repair_vs_objective_off",
     "gru_cue_repair", "gru_objective_off"),
    ("ssm_cue_repair_vs_objective_off",
     "ssm_cue_repair", "ssm_objective_off"),
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def _seed(base: int, key: str) -> int:
    digest = hashlib.sha256(key.encode()).digest()
    return (base + int.from_bytes(digest[:8], "big")) % (2**63 - 1)


def crossed_bootstrap_ci(matrix: np.ndarray, *, draws: int, seed: int,
                         confidence: float = 0.95) -> dict[str, float]:
    """Crossed bootstrap over checkpoint seeds and episodes.

    Each draw independently resamples rows and columns, then evaluates their
    Cartesian product.  Supplying a paired-difference matrix therefore keeps
    the candidate/reference pairing intact at both levels.
    """

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 2 \
            or not np.isfinite(values).all():
        raise ValueError("crossed bootstrap requires a finite 2-D matrix")
    rng = np.random.default_rng(seed)
    row_count, column_count = values.shape
    row_weights = rng.multinomial(
        row_count, np.full(row_count, 1.0 / row_count), size=draws)
    column_weights = rng.multinomial(
        column_count, np.full(column_count, 1.0 / column_count), size=draws)
    samples = np.einsum(
        "dr,rc,dc->d", row_weights, values, column_weights,
        optimize=True) / float(row_count * column_count)
    tail = (1.0 - confidence) / 2.0
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(samples, tail)),
        "ci_high": float(np.quantile(samples, 1.0 - tail)),
        "confidence": confidence,
        "draws": int(draws),
    }


def _validate_metric_record(metrics: Mapping[str, Any], labels: np.ndarray,
                            label: str) -> None:
    arrays = {
        key: np.asarray(metrics.get(key), dtype=np.float64)
        for key in ("correct", "success", "executed_return",
                    "regret_to_label_oracle", "distance")
    }
    if metrics.get("episodes") != len(labels) or any(
            value.shape != (len(labels),) or not np.isfinite(value).all()
            for value in arrays.values()):
        raise ValueError(f"invalid metric arrays for {label}")
    prediction = np.asarray(metrics.get("prediction"), dtype=np.int64)
    if prediction.shape != labels.shape or not np.isin(
            prediction, np.arange(4)).all():
        raise ValueError(f"invalid prediction for {label}")
    expected = {
        "goal_decision_accuracy": np.mean(prediction == labels),
        "executed_success_rate": arrays["success"].mean(),
        "mean_executed_return": arrays["executed_return"].mean(),
        "mean_regret_to_label_oracle": arrays[
            "regret_to_label_oracle"].mean(),
    }
    if not np.array_equal(arrays["correct"], prediction == labels) \
            or any(not np.isclose(metrics.get(key), value,
                                  rtol=1e-12, atol=1e-12)
                   for key, value in expected.items()):
        raise ValueError(f"metric scalar/array mismatch for {label}")


def _read_cell(path: Path, spec: Mapping[str, Any], task: str,
               seed: int) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read evaluation {path}: {error}") from error
    expected = {
        "study": spec["study"], "task": task,
        "checkpoint_seed": seed, "spec": spec["_spec_record"],
    }
    failed = [key for key, value in expected.items()
              if result.get(key) != value]
    expected_slugs = {source_slug(spec, source) for source in SOURCE_IDS}
    if failed or set(result.get("sources", {})) != expected_slugs:
        raise ValueError(f"evaluation identity/grid mismatch {path}: {failed}")
    labels = np.asarray(result.get("labels"), dtype=np.int64)
    if labels.shape != (240,) or not np.isin(labels, np.arange(4)).all():
        raise ValueError(f"invalid evaluation labels in {path}")
    if result.get("consumer", {}).get("source_order") != list(SOURCE_IDS) \
            or result.get("consumer", {}).get(
                "shared_across_all_sources") is not True:
        raise ValueError(f"consumer was not shared in {path}")
    digest = result.get("consumer", {}).get("digest")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"invalid shared-consumer digest in {path}")
    for source in SOURCE_IDS:
        record = result["sources"][source_slug(spec, source)]
        if record.get("id") != source:
            raise ValueError(f"source identity mismatch in {path}")
        metrics = record.get("metrics", {})
        _validate_metric_record(metrics, labels, f"{path}/{source}")
        for array_key in ARRAY_ENDPOINTS.values():
            values = np.asarray(metrics.get(array_key), dtype=np.float64)
            if values.shape != (240,) or not np.isfinite(values).all():
                raise ValueError(
                    f"invalid {source}/{array_key} array in {path}")
    controls = result.get("controls", {})
    shuffled = controls.get("label_shuffle", {})
    if set(shuffled) != expected_slugs:
        raise ValueError(f"label-shuffle grid mismatch in {path}")
    for source in SOURCE_IDS:
        slug = source_slug(spec, source)
        _validate_metric_record(
            shuffled[slug].get("metrics", {}), labels,
            f"{path}/label-shuffle/{source}")
    _validate_metric_record(
        controls.get("action_time", {}), labels, f"{path}/action-time")
    oracle = result.get("label_oracle", {})
    _validate_metric_record(oracle, labels, f"{path}/label-oracle")
    if not np.array_equal(np.asarray(oracle.get("prediction")), labels):
        raise ValueError(f"label oracle did not execute true choices in {path}")
    limit = float(spec["controls"]["shortcut_accuracy_max"])
    max_shuffle = max(
        shuffled[slug]["metrics"]["goal_decision_accuracy"]
        for slug in expected_slugs)
    expected_validity = {
        "label_oracle_success_pass": (
            oracle["executed_success_rate"] >=
            float(spec["executed_choice"]["oracle_success_min"])),
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
        raise ValueError(f"validity receipt mismatch in {path}")
    permutation = np.asarray(
        controls.get("training_episode_permutation"), dtype=np.int64)
    if permutation.shape != (1200,) or not np.array_equal(
            np.sort(permutation), np.arange(1200)):
        raise ValueError(f"label-shuffle permutation mismatch in {path}")
    pairing = result.get("repair_pairing", {})
    if set(pairing) != {"gru", "ssm"} or any(
            record.get("shared_head_initialization") is not True
            or record.get("shared_cuda_device") is not True
            or not isinstance(
                record.get("repair_head_initial_state_sha256"), str)
            or len(record["repair_head_initial_state_sha256"]) != 64
            or not isinstance(record.get("cuda_device"), list)
            or len(record["cuda_device"]) != 2
            for record in pairing.values()):
        raise ValueError(f"repair-pair receipt mismatch in {path}")
    return result


def _matrix(cells: list[dict[str, Any]], spec: Mapping[str, Any],
            source: str, array_key: str) -> np.ndarray:
    slug = source_slug(spec, source)
    return np.stack([
        np.asarray(cell["sources"][slug]["metrics"][array_key],
                   dtype=np.float64)
        for cell in cells
    ])


def _task_summary(spec: Mapping[str, Any], task: str,
                  cells: list[dict[str, Any]]) -> dict[str, Any]:
    labels = np.asarray(cells[0]["labels"], dtype=np.int64)
    if any(not np.array_equal(labels, np.asarray(cell["labels"]))
           for cell in cells[1:]):
        raise ValueError(f"validation episode order differs across {task} seeds")
    bootstrap = spec["endpoints"]["bootstrap"]
    draws = int(bootstrap["draws"])
    base_seed = int(bootstrap["seed"])
    sources: dict[str, Any] = {}
    matrices: dict[str, dict[str, np.ndarray]] = {}
    for source in SOURCE_IDS:
        source_matrices = {
            endpoint: _matrix(cells, spec, source, array_key)
            for endpoint, array_key in ARRAY_ENDPOINTS.items()
        }
        matrices[source] = source_matrices
        sources[source_slug(spec, source)] = {
            "id": source,
            "name": next(record["name"]
                         for record in spec["representation_sources"]
                         if record["id"] == source),
            "endpoints": {
                endpoint: crossed_bootstrap_ci(
                    matrix, draws=draws,
                    seed=_seed(base_seed, f"{task}/{source}/{endpoint}"))
                for endpoint, matrix in source_matrices.items()
            },
        }

    contrasts: dict[str, Any] = {}
    scientific_validity = all(
        cell.get("validity", {}).get("valid_for_use_claim") is True
        for cell in cells)
    for name, candidate, reference in COMPARISONS:
        success = (matrices[candidate]["executed_success_rate"]
                   - matrices[reference]["executed_success_rate"])
        regret_reduction = (
            matrices[reference]["mean_regret_to_label_oracle"]
            - matrices[candidate]["mean_regret_to_label_oracle"])
        success_ci = crossed_bootstrap_ci(
            success, draws=draws,
            seed=_seed(base_seed, f"{task}/{name}/success"))
        regret_ci = crossed_bootstrap_ci(
            regret_reduction, draws=draws,
            seed=_seed(base_seed, f"{task}/{name}/regret"))
        contrasts[name] = {
            "candidate": candidate,
            "reference": reference,
            "executed_success_difference": success_ci,
            "regret_reduction": regret_ci,
            "both_lower_95pct_bounds_above_zero": (
                success_ci["ci_low"] > 0.0 and regret_ci["ci_low"] > 0.0),
            "valid_for_use_claim": scientific_validity,
            "use_success": bool(
                scientific_validity and success_ci["ci_low"] > 0.0
                and regret_ci["ci_low"] > 0.0),
        }

    shuffle_accuracy: dict[str, float] = {}
    for source in SOURCE_IDS:
        slug = source_slug(spec, source)
        shuffle_accuracy[slug] = float(np.mean([
            cell["controls"]["label_shuffle"][slug]["metrics"][
                "goal_decision_accuracy"] for cell in cells]))
    controls = {
        "mean_label_shuffle_accuracy_by_source": shuffle_accuracy,
        "mean_action_time_accuracy": float(np.mean([
            cell["controls"]["action_time"]["goal_decision_accuracy"]
            for cell in cells])),
        "all_cell_validity_gates_pass": scientific_validity,
        "failed_checkpoint_seeds": [
            cell["checkpoint_seed"] for cell in cells
            if cell.get("validity", {}).get("valid_for_use_claim") is not True],
    }
    return {
        "name": spec["tasks"][task]["name"],
        "slug": spec["tasks"][task]["slug"],
        "checkpoint_seeds": list(SEEDS),
        "validation_episodes": int(len(labels)),
        "sources": sources,
        "paired_contrasts": contrasts,
        "controls": controls,
    }


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.execute:
        raise SystemExit("refusing aggregation without explicit --execute")
    spec = load_locked_spec(args.spec)
    output = resolve_path(spec["output"]["summary"])
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    source_records: list[dict[str, Any]] = []
    tasks: dict[str, Any] = {}
    for task in TASKS:
        cells = []
        for seed in SEEDS:
            path = evaluation_directory(spec, task, seed) / "metrics.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing expected evaluation {path}")
            cells.append(_read_cell(path, spec, task, seed))
            source_records.append({
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
            })
        tasks[task] = _task_summary(spec, task, cells)
    summary = {
        "schema_version": 1,
        "study": spec["study"],
        "spec": spec["_spec_record"],
        "status": ("valid" if all(
            task["controls"]["all_cell_validity_gates_pass"]
            for task in tasks.values()) else "invalid_controls_or_oracle"),
        "inference": spec["endpoints"]["bootstrap"],
        "tasks": tasks,
        "source_evaluations": source_records,
        "claim_boundary": spec["claim_boundary"],
        "repair_interpretation": spec["repair"]["interpretation"],
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
    print(f"[delayed-goal-use] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
