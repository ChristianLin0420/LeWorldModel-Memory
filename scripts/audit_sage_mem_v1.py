#!/usr/bin/env python3
"""Independent grid, artifact, resource, and paired-inference audit for SAGE-Mem."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_sage_mem_v1 import (  # noqa: E402
    SageMemRunError, atomic_json, validate_cell_directory,
    validate_development_cell_directory,
)
from scripts.sage_mem_v1_spec import (  # noqa: E402
    ARMS, COHORTS, DEFAULT_SPEC, DEVELOPMENT_ARMS, DEVELOPMENT_SEEDS,
    FORMAL_SEEDS, cell_directory, development_cell_directory,
    development_cells, formal_cells, load_spec, output_root, sha256_file,
    spec_fingerprint,
)


REQUIRED_EPISODE_ARRAYS = (
    "episode_id", "class_id", "evidence_age", "retention_correct",
    "reset_correct", "exposure_correct", "next_feature_mse",
    "reset_next_feature_mse", "oracle_success", "execution_success",
)
TRAINABLE_PARAMETER_MATCHED_ARMS = tuple(
    arm for arm in ARMS if arm != "none")
BASELINE_ARMS = (
    "gru", "lstm", "ssm", "fixed_trust", "gdelta", "fixed_trust_aux",
    "ssm_aux",
)


class SageMemAuditError(RuntimeError):
    """Formal evidence is incomplete, mismatched, or violates the protocol."""


def _development_grid(spec: Mapping[str, Any]) -> dict[
        tuple[str, str, int], Mapping[str, Any]]:
    manifests = {}
    expected_paths = set()
    cohort_reference: dict[str, np.ndarray] = {}
    for cohort, arm, seed in development_cells(spec):
        path = development_cell_directory(spec, cohort, arm, seed)
        expected_paths.add(path.resolve())
        try:
            manifest = validate_development_cell_directory(
                spec, path, cohort, arm, seed)
        except (SageMemRunError, FileNotFoundError) as error:
            raise SageMemAuditError(
                f"development grid incomplete at {cohort}/{arm}/{seed}: "
                f"{error}") from error
        manifests[(cohort, arm, seed)] = manifest
        selected = int(spec["cohorts"][cohort][
            "split_episodes"]["development"])
        readout = selected - int(np.floor(0.75 * selected))
        if cohort == "dinowm_pointmaze_goal":
            readout *= 4
        artifact = path / manifest["result"]["episode_results"]["path"]
        arrays = _load_episode_arrays(
            artifact, classes=int(spec["cohorts"][cohort]["classes"]),
            ages=tuple(spec["cohorts"][cohort]["ages"]),
            episodes_per_age=readout, require_shared_episode_ids=False)
        identity = np.stack((arrays["episode_id"], arrays["class_id"],
                             arrays["evidence_age"]), axis=1)
        if cohort not in cohort_reference:
            cohort_reference[cohort] = identity
        elif not np.array_equal(identity, cohort_reference[cohort]):
            raise SageMemAuditError(
                f"cross-arm/seed development identity differs: "
                f"{cohort}/{arm}/{seed}")
    root = output_root(spec) / "development" / "cells"
    if root.exists():
        actual = {path.parent.resolve() for path in root.rglob("manifest.json")}
        unexpected = actual.difference(expected_paths)
        partial = [path for path in root.rglob(".*.partial-*") if path.is_dir()]
        if unexpected or partial:
            raise SageMemAuditError(
                f"unexpected/partial development cells: "
                f"{sorted(map(str, unexpected))}, "
                f"partial={sorted(map(str, partial))}")
    return manifests


def _select_comparator(values: Mapping[str, float], *, maximize: bool,
                       tie_break: list[str]) -> str:
    if not values:
        raise SageMemAuditError("no healthy development comparator remains")
    order = {arm: index for index, arm in enumerate(tie_break)}
    missing = set(values).difference(order)
    if missing:
        raise SageMemAuditError(
            f"development tie-break lacks arms: {sorted(missing)}")
    if maximize:
        return min(values, key=lambda arm: (-values[arm], order[arm]))
    return min(values, key=lambda arm: (values[arm], order[arm]))


def audit_development(spec: Mapping[str, Any]) -> tuple[
        dict[str, Any], dict[str, dict[str, Any]]]:
    manifests = _development_grid(spec)
    selections: dict[str, dict[str, Any]] = {}
    comparator_arms = list(spec["development_selection"]["comparator_arms"])
    tie_break = list(spec["development_selection"]["tie_break_order"])
    parameter_margin = float(
        spec["fairness_reporting"]["maximum_parameter_relative_gap"])
    gdelta_gate = spec["admission"]["gdelta_development_health_gate"]
    for cohort in COHORTS:
        target = int(spec["cohorts"][cohort]["target_parameters"])
        arm_summary: dict[str, Any] = {}
        for arm in DEVELOPMENT_ARMS:
            results = [manifests[(cohort, arm, seed)]["result"]
                       for seed in DEVELOPMENT_SEEDS]
            metrics = {
                name: float(np.mean([
                    result["development_metrics"][name]
                    for result in results]))
                for name in (
                    "next_feature_mse", "retention_balanced_accuracy",
                    "execution_success")
            }
            parameter_counts = {
                int(result["resource_report"]["trainable_parameters"])
                for result in results}
            if len(parameter_counts) != 1:
                raise SageMemAuditError(
                    f"development parameters vary: {cohort}/{arm}")
            parameters = parameter_counts.pop()
            parameter_gap = (0.0 if arm == "none" and parameters == 0 else
                             abs(parameters - target) / target)
            resource_means = {
                name: float(np.mean([
                    result["resource_report"][name] for result in results]))
                for name in spec["fairness_reporting"][
                    "required_per_arm_fields"]
                if name != "trainable_parameters"}
            healthy = (
                all(result["gradient_finite"] is True for result in results)
                and (arm == "none" or parameter_gap <= parameter_margin))
            arm_summary[arm] = {
                "seeds": list(DEVELOPMENT_SEEDS),
                "metrics": metrics,
                "trainable_parameters": parameters,
                "target_parameters": target,
                "parameter_relative_gap": parameter_gap,
                "resource_report_mean": resource_means,
                "finite_gradients_all_seeds": all(
                    result["gradient_finite"] is True for result in results),
                "healthy": healthy,
            }
        baseline_flops = [
            arm_summary[arm]["resource_report_mean"][
                "forward_flops_per_episode"]
            for arm in comparator_arms if arm != "gdelta"
            and arm_summary[arm]["healthy"]]
        if not baseline_flops or float(np.median(baseline_flops)) <= 0:
            raise SageMemAuditError(
                f"no positive healthy FLOP reference: {cohort}")
        median_flops = float(np.median(baseline_flops))
        flop_margin = float(
            spec["fairness_reporting"]["maximum_flop_relative_gap"])
        for arm in DEVELOPMENT_ARMS:
            flops = arm_summary[arm]["resource_report_mean"][
                "forward_flops_per_episode"]
            gap = abs(flops - median_flops) / median_flops
            arm_summary[arm]["flop_relative_gap_from_baseline_median"] = gap
            arm_summary[arm]["flop_matched"] = gap <= flop_margin
        reference_values = {
            arm: arm_summary[arm]["metrics"]["next_feature_mse"]
            for arm in gdelta_gate["reference_arms"]
            if arm_summary[arm]["healthy"]}
        if not reference_values:
            raise SageMemAuditError(
                f"no healthy gDelta reference arm: {cohort}")
        gdelta_mse = arm_summary["gdelta"]["metrics"]["next_feature_mse"]
        reference_mse = min(reference_values.values())
        gdelta_ratio = (1.0 if gdelta_mse == 0.0 and reference_mse == 0.0
                        else (float("inf") if reference_mse == 0.0
                              else gdelta_mse / reference_mse))
        gdelta_healthy = (
            arm_summary["gdelta"]["healthy"]
            and np.isfinite(gdelta_ratio)
            and gdelta_ratio <= gdelta_gate["next_feature_mse_ratio_max"])
        arm_summary["gdelta"]["next_feature_mse_ratio_to_best_reference"] = \
            (float(gdelta_ratio) if np.isfinite(gdelta_ratio) else None)
        arm_summary["gdelta"]["healthy"] = gdelta_healthy
        eligible = [arm for arm in comparator_arms
                    if arm_summary[arm]["healthy"]]
        retention_values = {
            arm: arm_summary[arm]["metrics"][
                "retention_balanced_accuracy"] for arm in eligible}
        next_values = {
            arm: arm_summary[arm]["metrics"]["next_feature_mse"]
            for arm in eligible}
        bank_path = (output_root(spec) / "development_banks" / cohort
                     / "manifest.json")
        retention_comparator = _select_comparator(
            retention_values, maximize=True, tie_break=tie_break)
        try:
            bank_display = str(bank_path.relative_to(ROOT))
        except ValueError:
            bank_display = str(bank_path)
        selections[cohort] = {
            "schema_version": 1, "study": "sage-mem-v1",
            "stage": "development-selection", "status": "selected",
            "cohort": cohort,
            "protocol_fingerprint": spec_fingerprint(spec),
            "development_bank": {
                "path": bank_display,
                "sha256": sha256_file(bank_path),
            },
            "registered_cells_verified": (
                len(DEVELOPMENT_ARMS) * len(DEVELOPMENT_SEEDS)),
            "arm_summary": arm_summary,
            "gdelta_development_healthy": gdelta_healthy,
            "locked_comparators": {
                "retention": retention_comparator,
                "next_feature": _select_comparator(
                    next_values, maximize=False, tie_break=tie_break),
                "execution": retention_comparator,
            },
            "execution_comparator_status": (
                "retention comparator reused; no development execution "
                "consumer was evaluated"),
            "labels_used_only_for_posthoc_selection_metrics": True,
            "formal_data_read": False,
            "formal_execution_started": False,
        }
    global_receipt = {
        "schema_version": 1, "study": "sage-mem-v1",
        "stage": "development-audit", "status": "complete",
        "protocol_fingerprint": spec_fingerprint(spec),
        "registered_cells_verified": len(development_cells(spec)),
        "cohorts": {
            cohort: {
                "status": selections[cohort]["status"],
                "gdelta_development_healthy": selections[cohort][
                    "gdelta_development_healthy"],
                "locked_comparators": selections[cohort][
                    "locked_comparators"],
            }
            for cohort in COHORTS
        },
        "parent_train_only": True,
        "formal_data_read": False,
        "formal_execution_started": False,
    }
    return global_receipt, selections


def write_development_audit(spec: Mapping[str, Any], *, resume: bool
                            ) -> dict[str, Any]:
    global_receipt, selections = audit_development(spec)
    selection_receipts = {}
    for cohort, value in selections.items():
        path = (output_root(spec) / "development" / "selections" / cohort
                / "receipt.json")
        if path.exists():
            existing = json.loads(path.read_text())
            if not resume or existing != value:
                raise FileExistsError(
                    f"development selection receipt exists or differs: {path}")
        else:
            atomic_json(path, value)
        selection_receipts[cohort] = {
            "path": str(path.relative_to(ROOT)),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    global_receipt["selection_receipts"] = selection_receipts
    destination = output_root(spec) / "development" / "audit_receipt.json"
    if destination.exists():
        existing = json.loads(destination.read_text())
        if not resume or existing != global_receipt:
            raise FileExistsError(
                f"development audit receipt exists or differs: {destination}")
    else:
        atomic_json(destination, global_receipt)
    return global_receipt


def paired_cluster_bootstrap(
        left: np.ndarray, right: np.ndarray, strata: np.ndarray, *,
        cluster_ids: np.ndarray | None = None,
        draws: int = 20_000, seed: int = 2026070821,
        confidence: float = 0.95) -> dict[str, Any]:
    """Paired seed×episode bootstrap, stratified by class/evidence age.

    ``left`` and ``right`` are shaped ``(formal_seed, native_episode)`` and
    aligned before subtraction. Resampling the paired difference preserves all
    registered counterfactual and arm pairing.
    """
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    strata = np.asarray(strata)
    if left.shape != right.shape or left.ndim != 2 or left.size == 0:
        raise SageMemAuditError("paired arrays must share a non-empty 2-D shape")
    if strata.shape != (left.shape[1],):
        raise SageMemAuditError("strata must have one value per native episode")
    if cluster_ids is None:
        cluster_ids = np.arange(left.shape[1], dtype=np.int64)
    cluster_ids = np.asarray(cluster_ids)
    if cluster_ids.shape != (left.shape[1],):
        raise SageMemAuditError(
            "cluster_ids must have one value per episode-age row")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise SageMemAuditError("bootstrap inputs must be finite")
    if not isinstance(draws, int) or draws < 1:
        raise SageMemAuditError("draws must be a positive integer")
    if not 0.0 < confidence < 1.0:
        raise SageMemAuditError("confidence must be in (0,1)")
    clusters = np.unique(cluster_ids)
    cluster_rows = {
        cluster: np.flatnonzero(cluster_ids == cluster) for cluster in clusters}
    profiles: dict[tuple[str, ...], list[Any]] = {}
    for cluster, rows in cluster_rows.items():
        profile = tuple(sorted(str(value) for value in strata[rows]))
        profiles.setdefault(profile, []).append(cluster)
    if not profiles or any(not values for values in profiles.values()):
        raise SageMemAuditError("empty cluster bootstrap stratum")
    delta = left - right
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    seed_count = delta.shape[0]
    for draw in range(draws):
        sampled_seeds = rng.integers(0, seed_count, size=seed_count)
        seed_values = np.empty(seed_count, dtype=np.float64)
        for slot, seed_index in enumerate(sampled_seeds):
            # Preserve the registered class×age mix within each sampled seed.
            group_values = []
            for profile_clusters in profiles.values():
                selected = rng.integers(
                    0, len(profile_clusters), size=len(profile_clusters))
                sampled_rows = np.concatenate([
                    cluster_rows[profile_clusters[index]] for index in selected])
                group_values.append(delta[seed_index, sampled_rows].mean())
            seed_values[slot] = np.mean(group_values)
        samples[draw] = seed_values.mean()
    alpha = (1.0 - confidence) / 2.0
    return {
        "point": float(np.mean([
            delta[:, np.concatenate([
                cluster_rows[cluster] for cluster in profile_clusters])].mean()
            for profile_clusters in profiles.values()])),
        "lower": float(np.quantile(samples, alpha)),
        "upper": float(np.quantile(samples, 1.0 - alpha)),
        "confidence": confidence,
        "draws": draws,
        "seed": seed,
        "resampling_unit": "formal seed and native episode cluster",
        "pairing_preserved": True,
        "samples": samples,
    }


def _load_episode_arrays(
        path: Path, *, classes: int, ages: tuple[int, ...],
        episodes_per_age: int, require_shared_episode_ids: bool
        ) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = set(REQUIRED_EPISODE_ARRAYS).difference(archive.files)
        unexpected = set(archive.files).difference(REQUIRED_EPISODE_ARRAYS)
        if missing or unexpected:
            raise SageMemAuditError(
                f"episode artifact schema mismatch missing={sorted(missing)} "
                f"unexpected={sorted(unexpected)}")
        result = {name: np.asarray(archive[name])
                  for name in REQUIRED_EPISODE_ARRAYS}
    length = result["episode_id"].size
    if length == 0 or any(value.ndim != 1 or value.size != length
                          for value in result.values()):
        raise SageMemAuditError("episode artifact arrays must be aligned 1-D")
    expected_length = episodes_per_age * len(ages)
    if length != expected_length:
        raise SageMemAuditError(
            f"episode artifact has {length} rows; expected {expected_length}")
    for name in ("episode_id", "class_id", "evidence_age"):
        if not np.issubdtype(result[name].dtype, np.integer):
            raise SageMemAuditError(f"identity array must be integer: {name}")
    if set(np.unique(result["evidence_age"]).tolist()) != set(ages):
        raise SageMemAuditError("episode artifact evidence ages differ")
    if (result["class_id"] < 0).any() \
            or (result["class_id"] >= classes).any():
        raise SageMemAuditError("episode artifact class ID leaves range")
    reference_ids = None
    reference_classes = None
    for age in ages:
        selected = np.flatnonzero(result["evidence_age"] == age)
        if selected.size != episodes_per_age:
            raise SageMemAuditError(
                f"age {age} has {selected.size} rows; expected {episodes_per_age}")
        ids = result["episode_id"][selected]
        labels = result["class_id"][selected]
        if np.unique(ids).size != episodes_per_age:
            raise SageMemAuditError(f"duplicate native episode at age {age}")
        if set(np.unique(labels).tolist()) != set(range(classes)):
            raise SageMemAuditError(f"age {age} omits a registered class")
        order = np.argsort(ids)
        ids, labels = ids[order], labels[order]
        if reference_ids is None:
            reference_ids, reference_classes = ids, labels
        elif require_shared_episode_ids and (
                not np.array_equal(ids, reference_ids)
                or not np.array_equal(labels, reference_classes)):
            raise SageMemAuditError(
                "native episode/class pairing differs across evidence ages")
    for name in REQUIRED_EPISODE_ARRAYS[3:]:
        if not np.isfinite(result[name]).all():
            raise SageMemAuditError(f"non-finite episode metric: {name}")
    for name in ("retention_correct", "reset_correct", "exposure_correct",
                 "oracle_success", "execution_success"):
        if not np.isin(result[name], (0, 1)).all():
            raise SageMemAuditError(f"binary episode metric malformed: {name}")
    if (result["next_feature_mse"] < 0).any() \
            or (result["reset_next_feature_mse"] < 0).any():
        raise SageMemAuditError("MSE metrics must be non-negative")
    return result


def _prepared_comparators(spec: Mapping[str, Any], cohort: str) -> Mapping[str, str]:
    path = output_root(spec) / "receipts" / "prepare" / cohort / "receipt.json"
    if not path.is_file():
        raise SageMemAuditError(f"prepare receipt missing: {cohort}")
    value = json.loads(path.read_text())
    result = value.get("result", {})
    comparators = result.get("locked_comparators")
    if not isinstance(comparators, Mapping) or set(comparators) != {
            "retention", "next_feature", "execution"}:
        raise SageMemAuditError(f"development comparators not locked: {cohort}")
    if any(arm not in BASELINE_ARMS for arm in comparators.values()):
        raise SageMemAuditError(f"invalid locked comparator: {cohort}")
    return comparators


def _grid(spec: Mapping[str, Any]) -> tuple[
        dict[tuple[str, str, int], Mapping[str, Any]],
        dict[tuple[str, str, int], dict[str, np.ndarray]]]:
    manifests = {}
    arrays = {}
    expected_paths = set()
    cohort_reference: dict[str, np.ndarray] = {}
    for cohort, arm, seed in formal_cells(spec):
        path = cell_directory(spec, cohort, arm, seed)
        expected_paths.add(path.resolve())
        try:
            manifest = validate_cell_directory(
                spec, path, cohort, arm, seed)
        except (SageMemRunError, FileNotFoundError) as error:
            raise SageMemAuditError(
                f"formal grid incomplete at {cohort}/{arm}/{seed}: {error}") \
                from error
        key = (cohort, arm, seed)
        manifests[key] = manifest
        artifact = path / manifest["result"]["episode_results"]["path"]
        arrays[key] = _load_episode_arrays(
            artifact, classes=int(spec["cohorts"][cohort]["classes"]),
            ages=tuple(spec["cohorts"][cohort]["ages"]),
            episodes_per_age=int(spec["cohorts"][cohort][
                "split_episodes"]["formal_test"]),
            require_shared_episode_ids=True)
        identity = np.stack((
            arrays[key]["episode_id"], arrays[key]["class_id"],
            arrays[key]["evidence_age"]), axis=1)
        if cohort not in cohort_reference:
            cohort_reference[cohort] = identity
        elif not np.array_equal(identity, cohort_reference[cohort]):
            raise SageMemAuditError(
                f"cross-arm/seed formal identity differs: {cohort}/{arm}/{seed}")
    cells_root = output_root(spec) / "cells"
    if cells_root.exists():
        actual = {path.parent.resolve()
                  for path in cells_root.rglob("manifest.json")}
        unexpected = actual.difference(expected_paths)
        partial = list(cells_root.rglob(".seed-*.partial-*"))
        if unexpected or partial:
            raise SageMemAuditError(
                f"unexpected/partial cells: {sorted(map(str, unexpected))}, "
                f"partial={sorted(map(str, partial))}")
    return manifests, arrays


def _aligned_metric(arrays: Mapping[tuple[str, str, int], dict[str, np.ndarray]],
                    cohort: str, arm: str, metric: str
                    ) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    reference = None
    strata = None
    for seed in FORMAL_SEEDS:
        value = arrays[(cohort, arm, seed)]
        identity = np.stack((value["episode_id"], value["class_id"],
                             value["evidence_age"]), axis=1)
        if reference is None:
            reference = identity
            strata = np.asarray([
                f"{class_id}:{age}" for class_id, age in
                zip(value["class_id"], value["evidence_age"], strict=True)])
        elif not np.array_equal(identity, reference):
            raise SageMemAuditError(
                f"formal episode pairing differs across seeds: {cohort}/{arm}")
        rows.append(value[metric].astype(np.float64))
    return np.stack(rows), strata


def _audit_resource_parity(spec: Mapping[str, Any], cohort: str,
                           manifests: Mapping[tuple[str, str, int],
                                              Mapping[str, Any]]) -> dict[str, Any]:
    target = spec["cohorts"][cohort]["target_parameters"]
    margin = spec["fairness_reporting"]["maximum_parameter_relative_gap"]
    flop_margin = spec["fairness_reporting"]["maximum_flop_relative_gap"]
    result = {}
    reference_flops = []
    for arm in TRAINABLE_PARAMETER_MATCHED_ARMS:
        reports = [manifests[(cohort, arm, seed)]["result"]["resource_report"]
                   for seed in FORMAL_SEEDS]
        parameters = {int(report["trainable_parameters"])
                      for report in reports}
        if len(parameters) != 1:
            raise SageMemAuditError(f"parameter count varies: {cohort}/{arm}")
        count = parameters.pop()
        relative = abs(count - target) / target
        if relative > margin:
            raise SageMemAuditError(
                f"parameter mismatch: {cohort}/{arm} {count} vs {target}")
        flops = float(np.mean([
            report["forward_flops_per_episode"] for report in reports]))
        result[arm] = {
            "trainable_parameters": count, "target_parameters": target,
            "parameter_relative_gap": relative,
            "forward_flops_per_episode": flops,
            "persistent_state_floats": int(reports[0][
                "persistent_state_floats"]),
            "peak_cuda_bytes_mean": float(np.mean([
                report["peak_cuda_bytes"] for report in reports])),
            "wall_clock_train_seconds_mean": float(np.mean([
                report["wall_clock_train_seconds"] for report in reports])),
        }
        if arm in BASELINE_ARMS:
            reference_flops.append(flops)
    median_flops = float(np.median(reference_flops))
    for arm, record in result.items():
        record["flop_relative_gap_from_baseline_median"] = abs(
            record["forward_flops_per_episode"] - median_flops) / median_flops
        record["flop_matched"] = (
            record["flop_relative_gap_from_baseline_median"] <= flop_margin)
    return result


def audit(spec: Mapping[str, Any]) -> dict[str, Any]:
    manifests, arrays = _grid(spec)
    draws = int(spec["statistics"]["bootstrap_draws"])
    base_seed = int(spec["statistics"]["bootstrap_seed"])
    cohorts: dict[str, Any] = {}
    for cohort_index, cohort in enumerate(COHORTS):
        comparators = _prepared_comparators(spec, cohort)
        retention_reference = comparators["retention"]
        next_reference = comparators["next_feature"]
        execution_reference = comparators["execution"]
        full_retention, strata = _aligned_metric(
            arrays, cohort, "sage_mem_full", "retention_correct")
        cluster_ids = arrays[(cohort, "sage_mem_full", FORMAL_SEEDS[0])][
            "episode_id"]
        reference_retention, reference_strata = _aligned_metric(
            arrays, cohort, retention_reference, "retention_correct")
        if not np.array_equal(strata, reference_strata):
            raise SageMemAuditError(f"cross-arm pairing differs: {cohort}")
        exposure = paired_cluster_bootstrap(
            full_retention, reference_retention, strata,
            cluster_ids=cluster_ids, draws=draws,
            seed=base_seed + 100 * cohort_index)
        reset, _ = _aligned_metric(
            arrays, cohort, "sage_mem_full", "reset_correct")
        reset_effect = paired_cluster_bootstrap(
            full_retention, reset, strata, cluster_ids=cluster_ids, draws=draws,
            seed=base_seed + 100 * cohort_index + 1)
        full_mse, _ = _aligned_metric(
            arrays, cohort, "sage_mem_full", "next_feature_mse")
        reference_mse, _ = _aligned_metric(
            arrays, cohort, next_reference, "next_feature_mse")
        mse_difference = paired_cluster_bootstrap(
            full_mse, reference_mse, strata, cluster_ids=cluster_ids, draws=draws,
            seed=base_seed + 100 * cohort_index + 2, confidence=0.90)
        reference_mse_mean = float(reference_mse.mean())
        relative_mse_upper_raw = (
            0.0 if reference_mse_mean == 0.0
            and mse_difference["upper"] <= 0.0
            else (float("inf") if reference_mse_mean == 0.0
                  else mse_difference["upper"] / reference_mse_mean))
        relative_mse_upper = (float(relative_mse_upper_raw)
                              if np.isfinite(relative_mse_upper_raw) else None)
        reset_mse, _ = _aligned_metric(
            arrays, cohort, "sage_mem_full", "reset_next_feature_mse")
        full_mse_mean = float(full_mse.mean())
        reset_mse_mean = float(reset_mse.mean())
        reset_mse_ratio_raw = (
            1.0 if full_mse_mean == 0.0 and reset_mse_mean == 0.0
            else (float("inf") if full_mse_mean == 0.0
                  else reset_mse_mean / full_mse_mean))
        reset_mse_ratio = (float(reset_mse_ratio_raw)
                           if np.isfinite(reset_mse_ratio_raw) else None)
        controls = {}
        for offset, arm in enumerate((
                "sage_mem_next_only", "sage_mem_no_exposure",
                "sage_mem_exposure_only", "fixed_trust_aux", "ssm_aux"),
                start=3):
            control, _ = _aligned_metric(
                arrays, cohort, arm, "retention_correct")
            controls[arm] = paired_cluster_bootstrap(
                full_retention, control, strata, cluster_ids=cluster_ids,
                draws=draws,
                seed=base_seed + 100 * cohort_index + offset)
        oracle, _ = _aligned_metric(
            arrays, cohort, "sage_mem_full", "oracle_success")
        full_execution, _ = _aligned_metric(
            arrays, cohort, "sage_mem_full", "execution_success")
        reference_execution, _ = _aligned_metric(
            arrays, cohort, execution_reference, "execution_success")
        execution_eligible = float(oracle.mean()) >= \
            spec["confirmatory_gates"]["execution"]["oracle_gate"]
        execution = (paired_cluster_bootstrap(
            full_execution, reference_execution, strata,
            cluster_ids=cluster_ids, draws=draws,
            seed=base_seed + 100 * cohort_index + 8)
                     if execution_eligible else None)
        cohorts[cohort] = {
            "locked_comparators": dict(comparators),
            "resource_parity": _audit_resource_parity(
                spec, cohort, manifests),
            "next_feature_noninferior": (
                relative_mse_upper is not None
                and relative_mse_upper <= 0.02),
            "next_feature_relative_upper_90": relative_mse_upper,
            "host_output_exposure": {
                key: value for key, value in exposure.items()
                if key != "samples"},
            "host_output_exposure_pass": exposure["lower"] >= 0.05,
            "reset_effect": {key: value for key, value in reset_effect.items()
                             if key != "samples"},
            "reset_effect_pass": (
                reset_effect["lower"] >= 0.03
                and reset_mse_ratio is not None
                and reset_mse_ratio <= 1.25),
            "reset_to_full_mse_ratio": reset_mse_ratio,
            "mechanism_controls": {
                arm: {key: value for key, value in estimate.items()
                      if key != "samples"}
                for arm, estimate in controls.items()},
            "mechanism_controls_pass": all(
                estimate["lower"] >= 0.03 for estimate in controls.values()),
            "execution_eligible": execution_eligible,
            "execution": ({key: value for key, value in execution.items()
                           if key != "samples"}
                          if execution is not None else None),
            "execution_pass": (execution is not None
                               and execution["lower"] >= 0.03),
        }
    return {
        "schema_version": 1, "study": "sage-mem-v1", "stage": "audit",
        "status": "complete", "protocol_fingerprint": spec_fingerprint(spec),
        "formal_cells_verified": len(formal_cells(spec)),
        "bootstrap_draws_per_registered_contrast": draws,
        "cohorts": cohorts,
        "pooled_cross_host_score_computed": False,
        "universal_success_claim_permitted": False,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("development", "formal"),
                        default="formal")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    spec = load_spec(args.spec)
    if not args.execute:
        print(json.dumps({
            "study": "sage-mem-v1", "preview": True,
            "stage": args.stage,
            "development_cells_required": len(development_cells(spec)),
            "formal_cells_required": len(formal_cells(spec)),
            "bootstrap_draws": spec["statistics"]["bootstrap_draws"],
            "cohorts": list(COHORTS), "arms": list(ARMS),
            "no_files_read_or_written": True,
        }, sort_keys=True))
        return
    if args.stage == "development":
        result = write_development_audit(spec, resume=args.resume)
        print(json.dumps(result, sort_keys=True))
        return
    result = audit(spec)
    destination = output_root(spec) / "audit" / "receipt.json"
    if destination.exists():
        raise FileExistsError(f"audit receipt already exists: {destination}")
    atomic_json(destination, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
