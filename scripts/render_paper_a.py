#!/usr/bin/env python3
"""Render the expansion Paper A from its validated aggregate summary.

The renderer is intentionally a publication gate, not a second analysis
script.  Every reported result is read from ``summary.json``; the renderer
only formats registered aggregates and chooses conservative prose from the
sign of their confidence intervals.  It refuses partial grids, stale source
hashes, malformed summaries, and unknown or unfilled template placeholders.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import fmean
from typing import Any, Mapping, Sequence

try:
    from scripts.paper_a_appendix import render_appendix
except ModuleNotFoundError:  # direct ``python scripts/render_paper_a.py``
    from paper_a_appendix import render_appendix

try:
    import yaml
except ImportError as error:  # pragma: no cover - explicit dependency error
    raise SystemExit("render_paper_a.py requires PyYAML (import name 'yaml')") from error


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "outputs/paper_a_expansion/summary.json"
TEMPLATE_PATH = ROOT / "templates/PAPER_A.template.md"
OUTPUT_PATH = ROOT / "docs/PAPER_A.md"
MANIFEST_PATH = ROOT / "docs/PAPER_A.manifest.json"

EXPECTED_METRIC_FILES = 86
TASKS = ("t1", "t3")
ALL_TASKS = ("t1", "t3", "t4")
TASK_NAMES = {
    "t1": "Transient-marker recall",
    "t3": "Drifting-color recall",
    "t4": "Occluded-target prediction",
}
ARMS = ("none", "gru", "lstm", "ssm", "fixed_trust")
FROZEN_SEEDS = (0, 1, 2, 3, 4)
CONTEXTS = (3, 16, 32, 56)
MODEL_SEEDS = (0, 1, 2)
OBJECTIVES = ("one_step", "overshoot_8")
HORIZONS = (1, 2, 4, 8, 16)
ROLLOUT_METRICS = (
    "normalized_latent_mse",
    "copy_last_normalized_mse",
    "shuffled_action_normalized_mse",
    "true_action_advantage",
    "pose_angular_mae",
    "predicted_effective_rank",
    "target_effective_rank",
)

PLACEHOLDER_RE = re.compile(r"\{\{([A-Z0-9_]+)\}\}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
EXPECTED_PLACEHOLDERS = frozenset({
    "APPENDIX_BODY",
    "OCCLUDED_AVAILABILITY_R2",
    "POOLED_FIXED_TRUST_GRU_DIFFERENCE",
    "POOLED_FIXED_TRUST_GRU_CI_LOW",
    "POOLED_FIXED_TRUST_GRU_CI_HIGH",
    "ABSTRACT_CONTEXT_AND_ROLLOUT_FINDING",
    "MARKER_AVAILABILITY_ACC",
    "DRIFTING_AVAILABILITY_ACC",
    "MARKER_NONE_ACC_WITH_CI",
    "MARKER_FIXED_TRUST_ACC_WITH_CI",
    "DRIFTING_NONE_ACC_WITH_CI",
    "DRIFTING_FIXED_TRUST_ACC_WITH_CI",
    "MARKER_GRU_ACC_WITH_CI",
    "DRIFTING_GRU_ACC_WITH_CI",
    "POOLED_FIXED_TRUST_GRU_WINS",
    "POOLED_FIXED_TRUST_LSTM_DIFFERENCE_WITH_CI",
    "POOLED_FIXED_TRUST_SSM_DIFFERENCE_WITH_CI",
    "FROZEN_SWAP_INTERPRETATION",
    "NONE_MEAN_ACC_WITH_CI",
    "NONE_MEAN_NEXT_LATENT_MSE",
    "GRU_MEAN_ACC_WITH_CI",
    "GRU_MEAN_NEXT_LATENT_MSE",
    "MARKER_LSTM_ACC_WITH_CI",
    "DRIFTING_LSTM_ACC_WITH_CI",
    "LSTM_MEAN_ACC_WITH_CI",
    "LSTM_MEAN_NEXT_LATENT_MSE",
    "MARKER_SSM_ACC_WITH_CI",
    "DRIFTING_SSM_ACC_WITH_CI",
    "SSM_MEAN_ACC_WITH_CI",
    "SSM_MEAN_NEXT_LATENT_MSE",
    "FIXED_TRUST_MEAN_ACC_WITH_CI",
    "FIXED_TRUST_MEAN_NEXT_LATENT_MSE",
    "MARKER_RAW_CONTEXT_H3_ACC",
    "MARKER_RAW_CONTEXT_H56_ACC",
    "MARKER_PREDICTOR_H3_ACC_WITH_CI",
    "MARKER_PREDICTOR_H56_ACC_WITH_CI",
    "DRIFTING_RAW_CONTEXT_H3_ACC",
    "DRIFTING_RAW_CONTEXT_H56_ACC",
    "DRIFTING_PREDICTOR_H3_ACC_WITH_CI",
    "DRIFTING_PREDICTOR_H56_ACC_WITH_CI",
    "LONG_CONTEXT_INTERPRETATION",
    "MARKER_ONE_STEP_H8_MSE_RATIO",
    "DRIFTING_ONE_STEP_H8_MSE_RATIO",
    "MARKER_OVERSHOOT_H8_MSE_RATIO",
    "DRIFTING_OVERSHOOT_H8_MSE_RATIO",
    "MARKER_ONE_STEP_H8_ACTION_ADVANTAGE",
    "DRIFTING_ONE_STEP_H8_ACTION_ADVANTAGE",
    "MARKER_OVERSHOOT_H8_ACTION_ADVANTAGE",
    "DRIFTING_OVERSHOOT_H8_ACTION_ADVANTAGE",
    "MARKER_ONE_STEP_GATE_PASSES",
    "MARKER_OVERSHOOT_GATE_PASSES",
    "DRIFTING_ONE_STEP_GATE_PASSES",
    "DRIFTING_OVERSHOOT_GATE_PASSES",
    "ROLLOUT_GATE_INTERPRETATION",
    "PERSISTENCE_LOCALIZATION",
    "CONTEXT_CARRIER_LOCALIZATION",
    "DYNAMICS_LOCALIZATION",
    "MAIN_NO_CARRIER_ANALYSIS",
    "MAIN_TRAJECTORY_ANALYSIS",
    "MAIN_CONTEXT_MSE_ANALYSIS",
    "MAIN_ROLLOUT_TRADEOFF",
    "MAIN_TASK_HETEROGENEITY",
})


class RenderError(RuntimeError):
    """A fail-closed publication validation error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RenderError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _finite(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{label} must be a finite number",
    )
    return float(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _root_path(relative: Any, label: str) -> Path:
    _require(isinstance(relative, str) and relative, f"{label}.path is invalid")
    _require(not Path(relative).is_absolute(), f"{label}.path must be repository-relative")
    candidate = (ROOT / relative).resolve()
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError as error:
        raise RenderError(f"{label}.path leaves the repository: {relative}") from error
    return candidate


def _verify_hashed_record(record: Any, label: str) -> dict[str, str]:
    item = _mapping(record, label)
    path = _root_path(item.get("path"), label)
    digest = item.get("sha256")
    _require(isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None,
             f"{label}.sha256 is invalid")
    _require(path.is_file(), f"{label} source is missing: {item.get('path')}")
    actual = _sha256(path)
    _require(actual == digest, f"{label} source hash changed: {item.get('path')}")
    return {"path": str(item["path"]), "sha256": digest}


def _same_number(left: Any, right: Any) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def _validate_completion(summary: Mapping[str, Any]) -> None:
    completion = _mapping(summary.get("completion"), "summary.completion")
    completed = completion.get("completed_metrics")
    expected = completion.get("expected_metrics")
    if completion.get("complete") is not True:
        raise RenderError(
            "refusing to render from an incomplete expansion summary: "
            f"{completed}/{expected} metric files are complete"
        )
    _require(expected == EXPECTED_METRIC_FILES,
             f"expected an 86-cell preregistered grid, observed {expected}")
    _require(completed == EXPECTED_METRIC_FILES,
             f"expected 86 completed metric files, observed {completed}")
    _require(completion.get("missing_count") == 0 and completion.get("missing") == [],
             "complete summary still lists missing grid cells")
    _require(completion.get("allow_incomplete") is False,
             "publication summary was generated with --allow-incomplete")


def _validate_config(config: Mapping[str, Any]) -> None:
    _require(config.get("schema_version") == 1, "preregistration schema is not version 1")
    _require(config.get("display_names") == TASK_NAMES,
             "preregistration semantic task names changed")
    frozen = _mapping(config.get("frozen_carrier_swap"), "config.frozen_carrier_swap")
    _require(frozen.get("tasks") == list(TASKS), "frozen carrier task grid changed")
    _require(frozen.get("diagnostic_only_tasks") == ["t4"],
             "diagnostic-only task registration changed")
    _require(frozen.get("arms") == list(ARMS), "frozen carrier arm grid changed")
    _require(frozen.get("seeds") == list(FROZEN_SEEDS), "frozen carrier seeds changed")
    context = _mapping(config.get("long_context"), "config.long_context")
    _require(context.get("tasks") == list(TASKS), "long-context task grid changed")
    _require(context.get("contexts") == list(CONTEXTS), "long-context lengths changed")
    _require(context.get("seeds") == list(MODEL_SEEDS), "long-context seeds changed")
    rollout = _mapping(config.get("learned_rollout"), "config.learned_rollout")
    _require(rollout.get("tasks") == list(TASKS), "rollout task grid changed")
    _require(rollout.get("objectives") == list(OBJECTIVES), "rollout objectives changed")
    _require(rollout.get("seeds") == list(MODEL_SEEDS), "rollout seeds changed")
    _require(rollout.get("horizons") == list(HORIZONS), "rollout horizons changed")
    gates = _mapping(config.get("availability_gate"), "config.availability_gate")
    _require(_same_number(gates.get("categorical_accuracy_min"), 0.75),
             "categorical availability gate changed")
    _require(_same_number(gates.get("continuous_r2_min"), 0.30),
             "continuous availability gate changed")


def _validate_statistic(statistic: Any, label: str, expected_seeds: Sequence[int],
                        *, require_complete: bool = False) -> Mapping[str, Any]:
    stat = _mapping(statistic, label)
    seeds = stat.get("seeds")
    values = stat.get("values")
    expected = list(expected_seeds)
    _require(seeds == expected, f"{label}.seeds must be {expected}, observed {seeds}")
    _require(stat.get("n") == len(expected), f"{label}.n must be {len(expected)}")
    _require(isinstance(values, list) and len(values) == len(expected),
             f"{label}.values does not match its seeds")
    numeric = [_finite(value, f"{label}.values") for value in values]
    mean = _finite(stat.get("mean"), f"{label}.mean")
    _require(math.isclose(mean, fmean(numeric), rel_tol=1e-10, abs_tol=1e-12),
             f"{label}.mean does not equal the mean of values")
    ci = stat.get("ci95")
    _require(isinstance(ci, list) and len(ci) == 2, f"{label}.ci95 is invalid")
    low = _finite(ci[0], f"{label}.ci95[0]")
    high = _finite(ci[1], f"{label}.ci95[1]")
    _require(low <= high, f"{label}.ci95 is reversed")
    if require_complete:
        _require(stat.get("complete") is True, f"{label} is not marked complete")
        if "expected_seeds" in stat:
            _require(stat.get("expected_seeds") == expected,
                     f"{label}.expected_seeds does not match registration")
        else:
            _require(stat.get("expected_pairs") == len(expected),
                     f"{label}.expected_pairs does not match registration")
    return stat


def _expected_metric_paths() -> set[str]:
    paths: set[str] = set()
    for task in TASKS:
        for arm in ARMS:
            for seed in FROZEN_SEEDS:
                paths.add(
                    f"outputs/paper_a_expansion/frozen_swap/{task}/{arm}/s{seed}/metrics.json"
                )
        for history in CONTEXTS:
            for seed in MODEL_SEEDS:
                paths.add(
                    f"outputs/paper_a_expansion/long_context/{task}/h{history}/s{seed}/metrics.json"
                )
        for objective in OBJECTIVES:
            for seed in MODEL_SEEDS:
                paths.add(
                    f"outputs/paper_a_expansion/rollout/{task}/{objective}/s{seed}/metrics.json"
                )
    _require(len(paths) == EXPECTED_METRIC_FILES, "internal grid path count is not 86")
    return paths


def _validate_provenance(summary: Mapping[str, Any]) -> tuple[Mapping[str, Any], dict[str, Any]]:
    provenance = _mapping(summary.get("provenance"), "summary.provenance")
    config_record = _verify_hashed_record(provenance.get("config"), "preregistration")
    config_path = _root_path(config_record["path"], "preregistration")
    config_raw = yaml.safe_load(config_path.read_text())
    config = _mapping(config_raw, "preregistration YAML")
    _validate_config(config)

    _require(provenance.get("official_host") == config.get("official_host"),
             "summary official-host contract differs from preregistration")
    weights = _verify_hashed_record(
        provenance.get("official_host_weights"), "official host weights")
    registered_hash = _mapping(config.get("official_host"), "config.official_host").get(
        "weights_sha256")
    _require(weights["sha256"] == registered_hash,
             "official weights hash differs from preregistration")

    cache_records = _mapping(provenance.get("cache_manifests"),
                             "summary.provenance.cache_manifests")
    _require(set(cache_records) == set(ALL_TASKS),
             "cache-manifest provenance must cover exactly the three registered tasks")
    caches: dict[str, dict[str, str]] = {}
    for task in ALL_TASKS:
        item = _mapping(cache_records[task], f"cache manifest {task}")
        caches[task] = _verify_hashed_record(item, f"cache manifest {task}")
        checkpoint = _mapping(item.get("official_checkpoint"),
                              f"cache manifest {task}.official_checkpoint")
        _require(checkpoint.get("sha256") == registered_hash,
                 f"cache manifest {task} uses different host weights")
        _require(item.get("source_stream") == "clean",
                 f"cache manifest {task} is not the clean stream")

    source_hashes = _mapping(provenance.get("source_metric_sha256"),
                             "summary.provenance.source_metric_sha256")
    expected_paths = _expected_metric_paths()
    _require(set(source_hashes) == expected_paths,
             "source-metric ledger does not exactly match the 86-cell grid")
    verified_metrics: dict[str, str] = {}
    for relative in sorted(expected_paths):
        digest = source_hashes[relative]
        _require(isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None,
                 f"invalid source metric SHA-256: {relative}")
        path = _root_path(relative, f"source metric {relative}")
        _require(path.is_file(), f"missing source metric: {relative}")
        _require(_sha256(path) == digest, f"source metric hash changed: {relative}")
        verified_metrics[relative] = digest

    state_digest = provenance.get("frozen_host_state_sha256")
    _require(isinstance(state_digest, str) and SHA256_RE.fullmatch(state_digest) is not None,
             "frozen host state SHA-256 is missing or invalid")
    manifest_sources = {
        "preregistration": config_record,
        "official_weights": weights,
        "cache_manifests": caches,
        "metric_files": verified_metrics,
    }
    return config, manifest_sources


def _validate_availability(summary: Mapping[str, Any], config: Mapping[str, Any],
                           manifest_sources: dict[str, Any]) -> None:
    availability = _mapping(summary.get("availability"), "summary.availability")
    _require(set(availability) == set(ALL_TASKS),
             "availability summary must cover exactly the three registered tasks")
    gates = _mapping(config.get("availability_gate"), "config.availability_gate")
    expected = {
        "t1": ("accuracy", "carrier_ranking", gates["categorical_accuracy_min"], True),
        "t3": ("accuracy", "carrier_ranking", gates["categorical_accuracy_min"], True),
        "t4": ("r2", "diagnostic_only", gates["continuous_r2_min"], False),
    }
    source_records: dict[str, dict[str, str]] = {}
    for task, (metric, role, threshold, passed) in expected.items():
        item = _mapping(availability[task], f"availability.{task}")
        _require(item.get("task_id") == task, f"availability.{task} task identity changed")
        _require(item.get("display_name") == TASK_NAMES[task],
                 f"availability.{task} display name changed")
        _require(item.get("metric") == metric, f"availability.{task} metric changed")
        _require(item.get("role") == role, f"availability.{task} role changed")
        _require(_same_number(item.get("threshold"), threshold),
                 f"availability.{task} threshold changed")
        _require(item.get("passed") is passed,
                 f"availability.{task} pass/fail role is inconsistent")
        value = _finite(item.get("value"), f"availability.{task}.value")
        derived = value >= float(threshold)
        _require(derived is passed, f"availability.{task} gate does not match its value")
        source = _verify_hashed_record(
            {"path": item.get("source"), "sha256": item.get("source_sha256")},
            f"availability source {task}",
        )
        raw = _mapping(json.loads(_root_path(source["path"], f"availability source {task}").read_text()),
                       f"availability source {task} JSON")
        _require(raw.get("task") == task and raw.get("metric") == metric,
                 f"availability source {task} identity changed")
        _require(_same_number(raw.get("value"), value),
                 f"availability source {task} value differs from summary")
        source_records[task] = source
    manifest_sources["availability_sources"] = source_records


def _validate_frozen(summary: Mapping[str, Any]) -> None:
    frozen = _mapping(summary.get("frozen_carrier_swap"), "summary.frozen_carrier_swap")
    tasks = _mapping(frozen.get("tasks"), "frozen_carrier_swap.tasks")
    _require(set(tasks) == set(TASKS), "frozen summary task grid is incomplete")
    for task in TASKS:
        task_item = _mapping(tasks[task], f"frozen task {task}")
        _require(task_item.get("task_id") == task, f"frozen task {task} identity changed")
        _require(task_item.get("display_name") == TASK_NAMES[task],
                 f"frozen task {task} display name changed")
        arms = _mapping(task_item.get("arms"), f"frozen task {task}.arms")
        _require(set(arms) == set(ARMS), f"frozen task {task} arm grid is incomplete")
        for arm in ARMS:
            arm_item = _mapping(arms[arm], f"frozen task {task}.{arm}")
            _require(arm_item.get("arm_id") == arm,
                     f"frozen task {task}.{arm} identity changed")
            _validate_statistic(arm_item.get("accuracy"),
                                f"frozen task {task}.{arm}.accuracy", FROZEN_SEEDS)
            _validate_statistic(
                arm_item.get("trajectory_accuracy"),
                f"frozen task {task}.{arm}.trajectory_accuracy", FROZEN_SEEDS)
            _validate_statistic(arm_item.get("validation_next_latent_mse"),
                                f"frozen task {task}.{arm}.mse", FROZEN_SEEDS)
            parameters = arm_item.get("carrier_parameters")
            _require(isinstance(parameters, int) and parameters >= 0,
                     f"frozen task {task}.{arm} parameter count is invalid")
        contrasts = _mapping(task_item.get("paired_contrasts"),
                             f"frozen task {task}.paired_contrasts")
        _require(set(contrasts) == {"gru", "lstm", "ssm"},
                 f"frozen task {task} contrast grid is incomplete")
        for reference in ("gru", "lstm", "ssm"):
            contrast = _validate_statistic(
                contrasts[reference], f"frozen task {task}.vs_{reference}",
                FROZEN_SEEDS, require_complete=True)
            _require(contrast.get("expected_pairs") == len(FROZEN_SEEDS),
                     f"frozen task {task}.vs_{reference} pair count changed")
        versus_none = _mapping(task_item.get("paired_vs_no_carrier"),
                               f"frozen task {task}.paired_vs_no_carrier")
        _require(set(versus_none) == {"gru", "lstm", "ssm", "fixed_trust"},
                 f"frozen task {task} no-carrier contrasts are incomplete")
        for arm, contrast_raw in versus_none.items():
            _validate_statistic(contrast_raw, f"frozen task {task}.{arm}_vs_none",
                                FROZEN_SEEDS, require_complete=True)

    pooled_arms = _mapping(frozen.get("pooled_equal_task_arms"),
                           "frozen_carrier_swap.pooled_equal_task_arms")
    _require(set(pooled_arms) == set(ARMS), "pooled carrier arm grid is incomplete")
    for arm in ARMS:
        item = _mapping(pooled_arms[arm], f"pooled carrier {arm}")
        _require(item.get("arm_id") == arm, f"pooled carrier {arm} identity changed")
        for metric in ("accuracy", "trajectory_accuracy",
                       "validation_next_latent_mse"):
            stat = _validate_statistic(item.get(metric), f"pooled carrier {arm}.{metric}",
                                       FROZEN_SEEDS, require_complete=True)
            _require(stat.get("tasks") == list(TASKS),
                     f"pooled carrier {arm}.{metric} task weighting changed")
            _require(stat.get("task_weighting") == "equal",
                     f"pooled carrier {arm}.{metric} is not equal-task weighted")

    pooled = _mapping(frozen.get("pooled_equal_task_contrasts"),
                      "frozen_carrier_swap.pooled_equal_task_contrasts")
    _require(set(pooled) == {"gru", "lstm", "ssm"},
             "pooled carrier contrasts are incomplete")
    for reference in ("gru", "lstm", "ssm"):
        item = _mapping(pooled[reference], f"pooled contrast {reference}")
        _require(item.get("n_tasks") == len(TASKS) and item.get("tasks") == list(TASKS),
                 f"pooled contrast {reference} is not two-task weighted")
        _require(item.get("n_seed_pairs") == 10
                 and item.get("total_task_seed_pairs") == 10,
                 f"pooled contrast {reference} does not contain ten task-seed pairs")
        wins = item.get("positive_task_seed_wins")
        ties = item.get("ties")
        _require(isinstance(wins, int) and isinstance(ties, int)
                 and 0 <= wins <= 10 and 0 <= ties <= 10,
                 f"pooled contrast {reference} win ledger is invalid")
        _finite(item.get("mean"), f"pooled contrast {reference}.mean")
        ci = item.get("ci95")
        _require(isinstance(ci, list) and len(ci) == 2,
                 f"pooled contrast {reference}.ci95 is invalid")
        low = _finite(ci[0], f"pooled contrast {reference}.ci95[0]")
        high = _finite(ci[1], f"pooled contrast {reference}.ci95[1]")
        _require(low <= high, f"pooled contrast {reference}.ci95 is reversed")


def _validate_context(summary: Mapping[str, Any]) -> None:
    context = _mapping(summary.get("long_context"), "summary.long_context")
    tasks = _mapping(context.get("tasks"), "long_context.tasks")
    _require(set(tasks) == set(TASKS), "long-context summary task grid is incomplete")
    for task in TASKS:
        item = _mapping(tasks[task], f"long-context task {task}")
        _require(item.get("task_id") == task and item.get("display_name") == TASK_NAMES[task],
                 f"long-context task {task} identity changed")
        contexts = _mapping(item.get("contexts"), f"long-context task {task}.contexts")
        _require(set(contexts) == {str(value) for value in CONTEXTS},
                 f"long-context task {task} context grid is incomplete")
        for history in CONTEXTS:
            cell = _mapping(contexts[str(history)], f"long-context task {task}.H{history}")
            _require(cell.get("history") == history,
                     f"long-context task {task}.H{history} identity changed")
            _validate_statistic(cell.get("trained_predictor_semantic_accuracy"),
                                f"long-context task {task}.H{history}.accuracy", MODEL_SEEDS)
            _validate_statistic(cell.get("validation_next_latent_mse"),
                                f"long-context task {task}.H{history}.mse", MODEL_SEEDS)
            raw = _mapping(cell.get("raw_legal_context_readout"),
                           f"long-context task {task}.H{history}.raw")
            _require(raw.get("metric") == "accuracy" and raw.get("seed_independent") is True,
                     f"long-context task {task}.H{history} raw readout contract changed")
            _finite(raw.get("value"), f"long-context task {task}.H{history}.raw.value")
            _require(_same_number(raw.get("chance"), 0.25),
                     f"long-context task {task}.H{history} chance changed")
        comparisons = _mapping(item.get("paired_vs_short_context"),
                               f"long-context task {task}.paired_vs_short_context")
        _require(set(comparisons) == {"16", "32", "56"},
                 f"long-context task {task} paired comparisons are incomplete")
        for history in (16, 32, 56):
            comparison = _mapping(comparisons[str(history)],
                                  f"long-context task {task}.H{history}_vs_H3")
            _validate_statistic(comparison.get("trained_semantic_accuracy_delta"),
                                f"long-context task {task}.H{history}_vs_H3.accuracy",
                                MODEL_SEEDS, require_complete=True)
            _validate_statistic(comparison.get("validation_mse_delta"),
                                f"long-context task {task}.H{history}_vs_H3.mse",
                                MODEL_SEEDS, require_complete=True)


def _validate_rollout(summary: Mapping[str, Any]) -> None:
    rollout = _mapping(summary.get("learned_rollout"), "summary.learned_rollout")
    tasks = _mapping(rollout.get("tasks"), "learned_rollout.tasks")
    _require(set(tasks) == set(TASKS), "rollout summary task grid is incomplete")
    for task in TASKS:
        item = _mapping(tasks[task], f"rollout task {task}")
        _require(item.get("task_id") == task and item.get("display_name") == TASK_NAMES[task],
                 f"rollout task {task} identity changed")
        objectives = _mapping(item.get("objectives"), f"rollout task {task}.objectives")
        _require(set(objectives) == set(OBJECTIVES),
                 f"rollout task {task} objective grid is incomplete")
        for objective in OBJECTIVES:
            objective_item = _mapping(objectives[objective],
                                      f"rollout task {task}.{objective}")
            _require(objective_item.get("objective_id") == objective,
                     f"rollout task {task}.{objective} identity changed")
            horizons = _mapping(objective_item.get("horizons"),
                                f"rollout task {task}.{objective}.horizons")
            _require(set(horizons) == {str(value) for value in HORIZONS},
                     f"rollout task {task}.{objective} horizon grid is incomplete")
            for horizon in HORIZONS:
                metrics = _mapping(horizons[str(horizon)],
                                   f"rollout task {task}.{objective}.H{horizon}")
                expected_metrics = set(ROLLOUT_METRICS) | {"model_to_copy_ratio"}
                _require(set(metrics) == expected_metrics,
                         f"rollout task {task}.{objective}.H{horizon} metrics are incomplete")
                for metric in ROLLOUT_METRICS:
                    _validate_statistic(metrics[metric],
                                        f"rollout task {task}.{objective}.H{horizon}.{metric}",
                                        MODEL_SEEDS)
                ratio = _validate_statistic(
                    metrics["model_to_copy_ratio"],
                    f"rollout task {task}.{objective}.H{horizon}.model_to_copy_ratio",
                    MODEL_SEEDS,
                )
                numerator = metrics["normalized_latent_mse"]
                denominator = metrics["copy_last_normalized_mse"]
                _require(ratio["seeds"] == numerator["seeds"] == denominator["seeds"],
                         f"rollout task {task}.{objective}.H{horizon} ratio seeds differ")
                for seed, observed, top, bottom in zip(
                        ratio["seeds"], ratio["values"], numerator["values"],
                        denominator["values"], strict=True):
                    _require(float(bottom) > 0 and _same_number(
                        observed, float(top) / float(bottom)),
                        f"rollout task {task}.{objective}.H{horizon}.seed{seed} "
                        "model/copy ratio is inconsistent")
            gate = _mapping(objective_item.get("competence_gate_through_horizon_8"),
                            f"rollout task {task}.{objective}.gate")
            _require(gate.get("expected_seeds") == list(MODEL_SEEDS)
                     and gate.get("evaluated_seeds") == list(MODEL_SEEDS),
                     f"rollout task {task}.{objective} gate seed ledger changed")
            passed = gate.get("passed_seeds")
            count = gate.get("pass_count")
            _require(isinstance(passed, list) and all(seed in MODEL_SEEDS for seed in passed)
                     and isinstance(count, int) and count == len(passed),
                     f"rollout task {task}.{objective} gate counts are invalid")
            _require(gate.get("all_evaluated_seeds_pass") is (count == len(MODEL_SEEDS))
                     and gate.get("all_preregistered_seeds_pass") is (passed == list(MODEL_SEEDS)),
                     f"rollout task {task}.{objective} gate flags are inconsistent")
        paired = _mapping(item.get("paired_overshoot_minus_one_step"),
                          f"rollout task {task}.paired_objectives")
        _require(set(paired) == {str(value) for value in HORIZONS},
                 f"rollout task {task} paired horizon grid is incomplete")
        for horizon in HORIZONS:
            metrics = _mapping(paired[str(horizon)],
                               f"rollout task {task}.paired.H{horizon}")
            _require(set(metrics) == set(ROLLOUT_METRICS),
                     f"rollout task {task}.paired.H{horizon} metrics are incomplete")
            for metric in ROLLOUT_METRICS:
                _validate_statistic(metrics[metric],
                                    f"rollout task {task}.paired.H{horizon}.{metric}",
                                    MODEL_SEEDS, require_complete=True)


def _validate_summary(summary: Mapping[str, Any]) -> tuple[Mapping[str, Any], dict[str, Any]]:
    _validate_completion(summary)
    _require(summary.get("schema_version") == 1, "summary schema is not version 1")
    _require(summary.get("study") == "paper-a-expansion", "summary study identity changed")
    _require(summary.get("semantic_task_names") == TASK_NAMES,
             "summary semantic task names changed")
    validation = _mapping(summary.get("validation"), "summary.validation")
    expected_flags = {
        "fail_closed",
        "grid_complete",
        "all_discovered_cells_schema_and_provenance_valid",
        "official_host_file_hash_matches_preregistration",
        "frozen_host_unchanged_within_every_completed_cell",
        "frozen_host_state_consistent_across_completed_cells",
        "parameter_matching_ledger_consistent_across_completed_cells",
    }
    _require(expected_flags.issubset(validation), "summary validation flags are incomplete")
    false_flags = sorted(key for key, value in validation.items() if value is not True)
    _require(not false_flags, f"summary validation flags are not all true: {false_flags}")
    config, manifest_sources = _validate_provenance(summary)
    _validate_availability(summary, config, manifest_sources)
    _validate_frozen(summary)
    _validate_context(summary)
    _validate_rollout(summary)
    return config, manifest_sources


def _rounded(value: float, digits: int) -> float:
    return 0.0 if abs(value) < 0.5 * 10 ** (-digits) else value


def _fmt(value: Any, digits: int = 3, *, sign: bool = False) -> str:
    number = _rounded(_finite(value, "formatted scalar"), digits)
    return f"{number:+.{digits}f}" if sign else f"{number:.{digits}f}"


def _fmt_stat(statistic: Mapping[str, Any], digits: int = 3,
              *, sign: bool = False) -> str:
    low, high = statistic["ci95"]
    return (
        f"{_fmt(statistic['mean'], digits, sign=sign)} "
        f"[{_fmt(low, digits, sign=sign)}, {_fmt(high, digits, sign=sign)}]"
    )


def _ci_direction(statistic: Mapping[str, Any]) -> str:
    low, high = (_finite(value, "confidence interval") for value in statistic["ci95"])
    if low > 0:
        return "positive"
    if high < 0:
        return "negative"
    return "unresolved"


def _fixed_relation(reference: str, statistic: Mapping[str, Any]) -> str:
    direction = _ci_direction(statistic)
    name = "GRU" if reference == "gru" else "diagonal state-space carrier"
    if direction == "positive":
        return f"beats the {name} (the paired interval is wholly positive)"
    if direction == "negative":
        return f"trails the {name} (the paired interval is wholly negative)"
    return f"is statistically tied with the {name} at this resolution (the interval includes zero)"


def _raw_access_status(summary: Mapping[str, Any], task: str) -> str:
    raw = summary["long_context"]["tasks"][task]["contexts"]["56"][
        "raw_legal_context_readout"]
    value = float(raw["value"])
    chance = float(raw["chance"])
    threshold = float(summary["availability"][task]["threshold"])
    if value >= threshold:
        return f"clears the {_fmt(threshold)} availability criterion"
    if value > chance:
        return f"is above {_fmt(chance)} chance but remains below the {_fmt(threshold)} criterion"
    return f"does not exceed {_fmt(chance)} chance"


def _predictor_access_status(summary: Mapping[str, Any], task: str) -> str:
    stat = summary["long_context"]["tasks"][task]["contexts"]["56"][
        "trained_predictor_semantic_accuracy"]
    chance = float(summary["long_context"]["tasks"][task]["contexts"]["56"][
        "raw_legal_context_readout"]["chance"])
    low, high = (float(value) for value in stat["ci95"])
    if low > chance:
        return "is interval-resolved above chance"
    if high < chance:
        return "is interval-resolved below chance"
    return "is not interval-resolved above chance"


def _overshoot_status(summary: Mapping[str, Any], task: str) -> str:
    stat = summary["learned_rollout"]["tasks"][task][
        "paired_overshoot_minus_one_step"]["8"]["normalized_latent_mse"]
    direction = _ci_direction(stat)
    if direction == "negative":
        return "improves normalized MSE"
    if direction == "positive":
        return "worsens normalized MSE"
    return "does not yield an interval-resolved MSE improvement"


def _gate_text(summary: Mapping[str, Any], task: str, objective: str) -> str:
    gate = summary["learned_rollout"]["tasks"][task]["objectives"][objective][
        "competence_gate_through_horizon_8"]
    return f"{gate['pass_count']}/{len(gate['expected_seeds'])}"


def _h8_ratio(summary: Mapping[str, Any], task: str, objective: str) -> float:
    metrics = summary["learned_rollout"]["tasks"][task]["objectives"][objective][
        "horizons"]["8"]
    numerator = metrics["normalized_latent_mse"]
    denominator = metrics["copy_last_normalized_mse"]
    _require(numerator["seeds"] == denominator["seeds"],
             f"rollout {task}/{objective} H8 ratio seeds are not paired")
    ratios: list[float] = []
    for seed, top, bottom in zip(numerator["seeds"], numerator["values"],
                                 denominator["values"], strict=True):
        top_value = _finite(top, f"rollout {task}/{objective}/seed{seed} H8 MSE")
        bottom_value = _finite(bottom, f"rollout {task}/{objective}/seed{seed} copy-last")
        _require(bottom_value > 0,
                 f"rollout {task}/{objective}/seed{seed} has nonpositive copy-last MSE")
        ratios.append(top_value / bottom_value)
    return fmean(ratios)


def _narratives(summary: Mapping[str, Any]) -> dict[str, str]:
    frozen = summary["frozen_carrier_swap"]
    pooled_contrasts = frozen["pooled_equal_task_contrasts"]
    gru_relation = _fixed_relation("gru", pooled_contrasts["gru"])
    ssm_relation = _fixed_relation("ssm", pooled_contrasts["ssm"])
    frozen_swap = (
        f"Under the equal-task paired estimand, fixed-trust {gru_relation}; "
        f"it {ssm_relation}."
    )

    long_context = (
        f"At $H=56$, the raw window for {TASK_NAMES['t1']} "
        f"{_raw_access_status(summary, 't1')}, while its predictor output "
        f"{_predictor_access_status(summary, 't1')}; for {TASK_NAMES['t3']}, "
        f"the raw window {_raw_access_status(summary, 't3')}, while its predictor output "
        f"{_predictor_access_status(summary, 't3')}."
    )

    raw_clear = sum(
        float(summary["long_context"]["tasks"][task]["contexts"]["56"][
            "raw_legal_context_readout"]["value"])
        >= float(summary["availability"][task]["threshold"])
        for task in TASKS
    )
    predictor_exposed = sum(
        float(summary["long_context"]["tasks"][task]["contexts"]["56"][
            "trained_predictor_semantic_accuracy"]["ci95"][0])
        > float(summary["long_context"]["tasks"][task]["contexts"]["56"][
            "raw_legal_context_readout"]["chance"])
        for task in TASKS
    )
    gate_cells = sum(
        summary["learned_rollout"]["tasks"][task]["objectives"][objective][
            "competence_gate_through_horizon_8"]["all_preregistered_seeds_pass"]
        for task in TASKS for objective in OBJECTIVES
    )
    gate_passes = sum(
        summary["learned_rollout"]["tasks"][task]["objectives"][objective][
            "competence_gate_through_horizon_8"]["pass_count"]
        for task in TASKS for objective in OBJECTIVES
    )
    gate_runs = sum(
        len(summary["learned_rollout"]["tasks"][task]["objectives"][objective][
            "competence_gate_through_horizon_8"]["expected_seeds"])
        for task in TASKS for objective in OBJECTIVES
    )
    rollout_gate = (
        f"The through-eight criterion passes in {gate_cells}/4 task--objective "
        f"cells and {gate_passes}/{gate_runs} trained models. This establishes "
        "dynamics competence under the two references, not cue retention or "
        "memory-conditioned control."
    )
    abstract = (
        f"At $H=56$, raw context clears the availability criterion on {raw_clear}/2 tasks, "
        f"whereas predictor output is interval-resolved above chance on "
        f"{predictor_exposed}/2; rollout competence passes in {gate_cells}/4 "
        "task--objective cells. "
        f"Overshooting {_overshoot_status(summary, 't1')} on {TASK_NAMES['t1']} and "
        f"{_overshoot_status(summary, 't3')} on {TASK_NAMES['t3']}."
    )

    pooled_arms = frozen["pooled_equal_task_arms"]
    ssm_accuracy = pooled_arms["ssm"]["accuracy"]
    persistence = (
        "In the equal-task aggregate, the diagonal SSM has the lowest next-latent "
        f"MSE ({_fmt(pooled_arms['ssm']['validation_next_latent_mse']['mean'], 4)}) "
        f"while its final accuracy is only {_fmt_stat(ssm_accuracy)}, whose interval "
        "overlaps four-way chance. Better local prediction therefore does not "
        "establish robust delayed retention."
    )

    versus_none = []
    for task in TASKS:
        contrast = frozen["tasks"][task]["paired_vs_no_carrier"]["fixed_trust"]
        direction = _ci_direction(contrast)
        if direction == "positive":
            result = "is interval-resolved above no carrier"
        elif direction == "negative":
            result = "is interval-resolved below no carrier"
        else:
            result = "is not interval-resolved from no carrier"
        versus_none.append(f"{result} on {TASK_NAMES[task]}")
    context_carrier = (
        f"At $H=56$, raw access clears the pre-specified criterion on {raw_clear}/2 tasks and "
        f"predictor output is resolved above chance on {predictor_exposed}/2; at frozen "
        f"$H=3$, fixed-trust {versus_none[0]} and {versus_none[1]}."
    )

    dynamics = (
        "The rollout test begins from the local anchor $t=24$ and never evaluates "
        "retention of evidence preceding that anchor. Passing its copy-last and "
        "action-shuffle references therefore establishes local dynamics competence, "
        "not episode memory, planning, or control."
    )

    frozen_tasks = frozen["tasks"]
    no_carrier = (
        "Relative to no carrier, the GRU changes final accuracy by "
        f"{_fmt_stat(frozen_tasks['t1']['paired_vs_no_carrier']['gru'], sign=True)} "
        f"on {TASK_NAMES['t1']} and "
        f"{_fmt_stat(frozen_tasks['t3']['paired_vs_no_carrier']['gru'], sign=True)} "
        f"on {TASK_NAMES['t3']}; the diagonal SSM changes it by "
        f"{_fmt_stat(frozen_tasks['t1']['paired_vs_no_carrier']['ssm'], sign=True)} "
        "and "
        f"{_fmt_stat(frozen_tasks['t3']['paired_vs_no_carrier']['ssm'], sign=True)}. "
        "Fixed-trust reverses by task: "
        f"{_fmt_stat(frozen_tasks['t1']['paired_vs_no_carrier']['fixed_trust'], sign=True)} "
        "versus "
        f"{_fmt_stat(frozen_tasks['t3']['paired_vs_no_carrier']['fixed_trust'], sign=True)}. "
        "The recurrent paths therefore produce small, task-dependent changes, "
        "while absolute final accuracies remain close to four-way chance."
    )

    trajectory = (
        "The exploratory trajectory-average readout reveals a different localization. "
        f"For {TASK_NAMES['t1']}, fixed-trust changes from "
        f"{_fmt(frozen_tasks['t1']['arms']['fixed_trust']['accuracy']['mean'])} "
        "at the final causal endpoint to "
        f"{_fmt(frozen_tasks['t1']['arms']['fixed_trust']['trajectory_accuracy']['mean'])} "
        "with temporal aggregation, and the SSM changes from "
        f"{_fmt(frozen_tasks['t1']['arms']['ssm']['accuracy']['mean'])} to "
        f"{_fmt(frozen_tasks['t1']['arms']['ssm']['trajectory_accuracy']['mean'])}. "
        f"On {TASK_NAMES['t3']}, the corresponding trajectory values are "
        f"{_fmt(frozen_tasks['t3']['arms']['fixed_trust']['trajectory_accuracy']['mean'])} "
        "and "
        f"{_fmt(frozen_tasks['t3']['arms']['ssm']['trajectory_accuracy']['mean'])}. "
        "The aggregate post-cue trajectory is therefore more linearly decodable "
        "than the final feature for these carriers. Because temporal support and "
        "feature maps differ, this does not prove exposure at any individual state "
        "or decision-time memory (Appendix Figure \\ref{fig:app-probe})."
    )

    context_summary = summary["long_context"]["tasks"]
    context_sentences = []
    for task in TASKS:
        h32 = context_summary[task]["contexts"]["32"]
        h56 = context_summary[task]["contexts"]["56"]
        mse32 = float(h32["validation_next_latent_mse"]["mean"])
        mse56 = float(h56["validation_next_latent_mse"]["mean"])
        increase = 100.0 * (mse56 / mse32 - 1.0)
        context_sentences.append(
            f"{TASK_NAMES[task]} raw access rises from "
            f"{_fmt(h32['raw_legal_context_readout']['value'])} to "
            f"{_fmt(h56['raw_legal_context_readout']['value'])}, while local "
            f"MSE worsens by {increase:.0f}\\%"
        )
    context_mse = (
        "The $H=32$ to $H=56$ transition makes the dissociation explicit: "
        + "; ".join(context_sentences)
        + ". Selecting context by minimum local prediction error would therefore "
        "prefer a window in which the cue is unreachable (Appendix Figure "
        "\\ref{fig:app-context})."
    )

    rollout_tasks = summary["learned_rollout"]["tasks"]
    marker_k16 = rollout_tasks["t1"]["paired_overshoot_minus_one_step"]["16"]
    color_k16 = rollout_tasks["t3"]["paired_overshoot_minus_one_step"]["16"]
    rollout_tradeoff = (
        "At $K=16$, overshooting reduces normalized latent MSE on "
        f"{TASK_NAMES['t1']} (paired difference "
        f"{_fmt_stat(marker_k16['normalized_latent_mse'], 4, sign=True)}) but "
        "increases pose MAE ("
        f"{_fmt_stat(marker_k16['pose_angular_mae'], 4, sign=True)}). On "
        f"{TASK_NAMES['t3']}, both paired differences indicate improvement: "
        f"{_fmt_stat(color_k16['normalized_latent_mse'], 4, sign=True)} for "
        "latent MSE and "
        f"{_fmt_stat(color_k16['pose_angular_mae'], 4, sign=True)} for pose MAE. "
        "Thus a latent-MSE improvement is not a uniform proxy for physical-state "
        "accuracy (Appendix Figure \\ref{fig:app-rollout})."
    )
    task_heterogeneity = (
        "The cross-task reversal is itself diagnostic. Fixed-trust is "
        f"{_fmt_stat(frozen_tasks['t1']['paired_vs_no_carrier']['fixed_trust'], sign=True)} "
        "relative to no carrier on transient-marker recall but "
        f"{_fmt_stat(frozen_tasks['t3']['paired_vs_no_carrier']['fixed_trust'], sign=True)} "
        "on drifting-color recall, whereas the SSM is positive on both tasks. "
        "The exploratory trajectory-average feature changes the ordering again. "
        "Carrier rankings therefore vary across tasks and endpoints, and cannot be "
        "explained by parameter count alone."
    )
    return {
        "ABSTRACT_CONTEXT_AND_ROLLOUT_FINDING": abstract,
        "FROZEN_SWAP_INTERPRETATION": frozen_swap,
        "LONG_CONTEXT_INTERPRETATION": long_context,
        "ROLLOUT_GATE_INTERPRETATION": rollout_gate,
        "PERSISTENCE_LOCALIZATION": persistence,
        "CONTEXT_CARRIER_LOCALIZATION": context_carrier,
        "DYNAMICS_LOCALIZATION": dynamics,
        "MAIN_NO_CARRIER_ANALYSIS": no_carrier,
        "MAIN_TRAJECTORY_ANALYSIS": trajectory,
        "MAIN_CONTEXT_MSE_ANALYSIS": context_mse,
        "MAIN_ROLLOUT_TRADEOFF": rollout_tradeoff,
        "MAIN_TASK_HETEROGENEITY": task_heterogeneity,
    }


def replacements(summary: Mapping[str, Any],
                 config: Mapping[str, Any]) -> dict[str, str]:
    availability = summary["availability"]
    frozen = summary["frozen_carrier_swap"]
    tasks = frozen["tasks"]
    pooled_arms = frozen["pooled_equal_task_arms"]
    pooled = frozen["pooled_equal_task_contrasts"]
    context = summary["long_context"]["tasks"]
    rollout = summary["learned_rollout"]["tasks"]

    values: dict[str, str] = {
        "MARKER_AVAILABILITY_ACC": _fmt(availability["t1"]["value"]),
        "DRIFTING_AVAILABILITY_ACC": _fmt(availability["t3"]["value"]),
        "OCCLUDED_AVAILABILITY_R2": _fmt(availability["t4"]["value"]),
        "POOLED_FIXED_TRUST_GRU_DIFFERENCE": _fmt(pooled["gru"]["mean"], sign=True),
        "POOLED_FIXED_TRUST_GRU_CI_LOW": _fmt(pooled["gru"]["ci95"][0], sign=True),
        "POOLED_FIXED_TRUST_GRU_CI_HIGH": _fmt(pooled["gru"]["ci95"][1], sign=True),
        "POOLED_FIXED_TRUST_GRU_WINS": (
            f"{pooled['gru']['positive_task_seed_wins']}/"
            f"{pooled['gru']['total_task_seed_pairs']}"
        ),
        "POOLED_FIXED_TRUST_LSTM_DIFFERENCE_WITH_CI": _fmt_stat(
            pooled["lstm"], sign=True),
        "POOLED_FIXED_TRUST_SSM_DIFFERENCE_WITH_CI": _fmt_stat(
            pooled["ssm"], sign=True),
    }

    task_prefix = {"t1": "MARKER", "t3": "DRIFTING"}
    arm_prefix = {
        "none": "NONE",
        "gru": "GRU",
        "lstm": "LSTM",
        "ssm": "SSM",
        "fixed_trust": "FIXED_TRUST",
    }
    for task, prefix in task_prefix.items():
        for arm, arm_name in arm_prefix.items():
            statistic = tasks[task]["arms"][arm]["accuracy"]
            values[f"{prefix}_{arm_name}_ACC_WITH_CI"] = (
                _fmt(statistic["mean"]) if arm == "none"
                else _fmt_stat(statistic)
            )
    for arm, prefix in arm_prefix.items():
        statistic = pooled_arms[arm]["accuracy"]
        values[f"{prefix}_MEAN_ACC_WITH_CI"] = (
            _fmt(statistic["mean"]) if arm == "none"
            else _fmt_stat(statistic)
        )
        values[f"{prefix}_MEAN_NEXT_LATENT_MSE"] = _fmt(
            pooled_arms[arm]["validation_next_latent_mse"]["mean"], 4)

    for task, prefix in task_prefix.items():
        values[f"{prefix}_RAW_CONTEXT_H3_ACC"] = _fmt(
            context[task]["contexts"]["3"]["raw_legal_context_readout"]["value"])
        values[f"{prefix}_RAW_CONTEXT_H56_ACC"] = _fmt(
            context[task]["contexts"]["56"]["raw_legal_context_readout"]["value"])
        values[f"{prefix}_PREDICTOR_H3_ACC_WITH_CI"] = _fmt_stat(
            context[task]["contexts"]["3"]["trained_predictor_semantic_accuracy"])
        values[f"{prefix}_PREDICTOR_H56_ACC_WITH_CI"] = _fmt_stat(
            context[task]["contexts"]["56"]["trained_predictor_semantic_accuracy"])

        for objective, objective_prefix in (("one_step", "ONE_STEP"),
                                             ("overshoot_8", "OVERSHOOT")):
            values[f"{prefix}_{objective_prefix}_H8_MSE_RATIO"] = _fmt(
                _h8_ratio(summary, task, objective))
            h8 = rollout[task]["objectives"][objective]["horizons"]["8"]
            values[f"{prefix}_{objective_prefix}_H8_ACTION_ADVANTAGE"] = _fmt(
                h8["true_action_advantage"]["mean"])
            values[f"{prefix}_{objective_prefix}_GATE_PASSES"] = _gate_text(
                summary, task, objective)

    values.update(_narratives(summary))
    values["APPENDIX_BODY"] = render_appendix(summary, config)
    _require(set(values) == EXPECTED_PLACEHOLDERS,
             "renderer replacement map does not exactly match its 56-placeholder contract")
    _require(all(PLACEHOLDER_RE.search(value) is None
                 for value in values.values()),
             "a replacement value contains a nested manuscript placeholder")
    return values


def _render_template(template: str, values: Mapping[str, str]) -> str:
    tokens = PLACEHOLDER_RE.findall(template)
    present = set(tokens)
    unknown = present - EXPECTED_PLACEHOLDERS
    missing = EXPECTED_PLACEHOLDERS - present
    _require(not unknown, f"unknown template placeholders: {sorted(unknown)}")
    _require(not missing, f"template is missing placeholders: {sorted(missing)}")

    # Replace in one regex pass.  A repeated semantic placeholder may occur in
    # several manuscript locations, but each concrete token occurrence is
    # consumed exactly once by this callback.
    expected_counts = Counter(tokens)
    replaced_counts: Counter[str] = Counter()

    def substitute(match: re.Match[str]) -> str:
        key = match.group(1)
        replaced_counts[key] += 1
        return values[key]

    rendered = PLACEHOLDER_RE.sub(substitute, template)
    _require(replaced_counts == expected_counts,
             "not every placeholder occurrence was filled exactly once")
    leftovers = re.findall(r"\{\{[^{}]*\}\}", rendered)
    _require(not leftovers, f"unfilled or malformed placeholders remain: {leftovers}")
    return rendered


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def main() -> None:
    try:
        _require(SUMMARY_PATH.is_file(), f"missing expansion summary: {SUMMARY_PATH}")
        summary_text = SUMMARY_PATH.read_text()
        summary = _mapping(json.loads(summary_text), "expansion summary")

        # Completion is checked first so a progress summary can never reach
        # template inspection or overwrite an existing manuscript.
        _validate_completion(summary)
        config, manifest_sources = _validate_summary(summary)

        _require(TEMPLATE_PATH.is_file(), f"missing canonical template: {TEMPLATE_PATH}")
        template = TEMPLATE_PATH.read_text()
        values = replacements(summary, config)
        manuscript = _render_template(template, values)

        summary_record = {
            "path": str(SUMMARY_PATH.relative_to(ROOT)),
            "sha256": _sha256(SUMMARY_PATH),
        }
        template_record = {
            "path": str(TEMPLATE_PATH.relative_to(ROOT)),
            "sha256": _sha256(TEMPLATE_PATH),
        }
        manuscript_digest = hashlib.sha256(manuscript.encode()).hexdigest()
        manifest = {
            "schema_version": 2,
            "manuscript": {
                "path": str(OUTPUT_PATH.relative_to(ROOT)),
                "sha256": manuscript_digest,
            },
            "sources": {
                "template": template_record,
                "summary": summary_record,
                **manifest_sources,
            },
        }
        _atomic_write(OUTPUT_PATH, manuscript)
        _atomic_write(MANIFEST_PATH,
                      json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(
            f"[render-a] wrote {OUTPUT_PATH} "
            f"({len(EXPECTED_PLACEHOLDERS)} result fields; "
            f"{len(PLACEHOLDER_RE.findall(template))} token occurrences)"
        )
    except (json.JSONDecodeError, yaml.YAMLError, RenderError) as error:
        raise SystemExit(f"[render-a] REFUSED: {error}") from error


if __name__ == "__main__":
    main()
