#!/usr/bin/env python3
"""Strict, fail-closed SAGE-Mem v1 protocol loader and cell planner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "configs/sage_mem_v1.yaml"
COHORTS = (
    "lewm_reacher_color",
    "lewm_pusht_color",
    "dinowm_pusht_token",
    "dinowm_pusht_binding",
    "dinowm_pointmaze_goal",
)
ARMS = (
    "none", "gru", "lstm", "ssm", "fixed_trust", "gdelta",
    "fixed_trust_aux", "ssm_aux", "sage_mem_full",
    "sage_mem_next_only", "sage_mem_no_exposure",
    "sage_mem_exposure_only",
)
FORMAL_SEEDS = tuple(range(10))
DEVELOPMENT_SEEDS = (101, 102, 103)
DEVELOPMENT_ARMS = ARMS
AGES = (4, 8, 15)
GPU_OWNERSHIP = {0: COHORTS[:2], 1: COHORTS[2:4], 2: (COHORTS[4],)}
STAGES = ("preflight", "smoke", "development", "seal", "prepare", "full",
          "audit")


class SageMemSpecError(ValueError):
    """The preregistered SAGE-Mem v1 contract is malformed or changed."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_repo_path(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise SageMemSpecError("path must be a non-empty repository-relative string")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise SageMemSpecError(f"path leaves repository: {value!r}")
    result = (ROOT / relative).resolve()
    try:
        result.relative_to(ROOT.resolve())
    except ValueError as error:
        raise SageMemSpecError(f"path leaves repository: {value!r}") from error
    return result


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise SageMemSpecError(f"{label} must be a mapping")
    return value


def _exact_list(value: Any, expected: tuple[Any, ...], label: str) -> None:
    if value != list(expected):
        raise SageMemSpecError(f"{label} must be exactly {list(expected)!r}")


def derive_seed(master_seed: int, cohort: str, split: str, purpose: str) -> int:
    """Derive a stable, namespaced 31-bit seed without Python hash()."""
    if cohort not in COHORTS:
        raise SageMemSpecError(f"unknown cohort: {cohort}")
    if not split or not purpose:
        raise SageMemSpecError("split and purpose must be non-empty")
    payload = "\0".join(
        ("sage-mem-v1", str(int(master_seed)), cohort, split, purpose)
    ).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value % 2_147_483_647


def seed_registry(spec: Mapping[str, Any]) -> dict[str, int]:
    master = int(_mapping(spec.get("freshness"), "freshness")["master_seed"])
    result: dict[str, int] = {}
    for cohort in COHORTS:
        for split in ("development", "formal_train", "consumer_train",
                      "formal_test"):
            for purpose in ("episode_selection", "cue_labels", "loader"):
                key = f"{cohort}/{split}/{purpose}"
                result[key] = derive_seed(master, cohort, split, purpose)
    if len(set(result.values())) != len(result):
        raise SageMemSpecError("derived seed collision")
    return result


def validate_spec(spec: Mapping[str, Any], *, verify_parent_paths: bool = False) -> None:
    if spec.get("schema_version") != 1 or spec.get("study") != "sage-mem-v1":
        raise SageMemSpecError("study identity changed")
    if spec.get("protocol_status") != \
            "preregistered-before-development-or-formal-outcomes":
        raise SageMemSpecError("protocol status changed")
    if spec.get("claim_scope") != \
            "task-level causal memory mechanism audit; no pooled cross-host score":
        raise SageMemSpecError("claim scope changed")
    if spec.get("implementation_lock") != \
            "outputs/sage_mem_v1/protocol_lock.json":
        raise SageMemSpecError("implementation lock path changed")

    cohorts = _mapping(spec.get("cohorts"), "cohorts")
    if tuple(cohorts) != COHORTS:
        raise SageMemSpecError(f"cohorts must be exactly {COHORTS!r}")
    expected_gpus = (0, 0, 1, 1, 2)
    expected_targets = (76_032, 76_032, 299_520, 299_520, 299_520)
    for index, cohort in enumerate(COHORTS):
        record = _mapping(cohorts[cohort], f"cohorts.{cohort}")
        if record.get("gpu") != expected_gpus[index]:
            raise SageMemSpecError(f"GPU ownership changed for {cohort}")
        if record.get("target_parameters") != expected_targets[index]:
            raise SageMemSpecError(f"parameter target changed for {cohort}")
        if record.get("ages") != list(AGES):
            raise SageMemSpecError(f"evidence ages changed for {cohort}")
        splits = _mapping(record.get("split_episodes"), f"{cohort}.splits")
        if tuple(splits) != ("development", "formal_train", "consumer_train",
                            "formal_test") or any(
                                not isinstance(value, int) or value <= 0
                                for value in splits.values()):
            raise SageMemSpecError(f"invalid split registry for {cohort}")
        forbidden = record.get("forbidden_parent_artifacts")
        if not isinstance(forbidden, list) or not forbidden:
            raise SageMemSpecError(f"missing parent exclusions for {cohort}")
        resolve_repo_path(record.get("parent_protocol"))
        for path in forbidden:
            resolved = resolve_repo_path(path)
            if verify_parent_paths and not resolved.is_file():
                raise SageMemSpecError(f"parent exclusion missing: {path}")

    _exact_list(spec.get("arms"), ARMS, "arms")
    model_interface = _mapping(spec.get("model_interface"), "model_interface")
    if model_interface != {
            "module": "lewm.models.sage_mem",
            "api_version": "sage_mem_v1_api_v1",
            "builder": "build_sage_mem_v1",
            "required_output_keys": [
                "fused", "prior", "posterior", "exposure", "diagnostics"],
            "labels_forbidden_in_builder_or_forward": True}:
        raise SageMemSpecError("model interface changed")
    host_interface = _mapping(
        spec.get("host_adapter_interface"), "host_adapter_interface")
    if host_interface != {
            "module": "scripts.sage_mem_v1_host_adapters",
            "api_version": "sage_mem_v1_host_adapter_v1",
            "builder": "build_host_adapter",
            "integration_status": (
                "development_adapter_delivered_formal_fresh_bank_pending")}:
        raise SageMemSpecError("host adapter interface changed")
    objective = _mapping(spec.get("label_free_objective"),
                         "label_free_objective")
    if objective != {
        "next_feature_mse_weight": 1.0,
        "exposure_alignment_weight": 0.10,
        "past_feature_replay_weight": 0.10,
        "targets": "frozen host features only",
        "semantic_labels_or_oracle_state_forbidden": True,
        "detach_all_targets": True,
        "objective_control_sets_auxiliary_weights_to_zero": True,
    }:
        raise SageMemSpecError("label-free objective changed")
    optimization = _mapping(spec.get("optimization"), "optimization")
    _exact_list(optimization.get("development_arms"), DEVELOPMENT_ARMS,
                "development_arms")
    _exact_list(optimization.get("development_seeds"), DEVELOPMENT_SEEDS,
                "development_seeds")
    _exact_list(optimization.get("formal_seeds"), FORMAL_SEEDS,
                "formal_seeds")
    if optimization.get("outcome_dependent_early_stopping") is not False \
            or optimization.get("equal_hyperparameter_budget_per_trainable_arm") \
            is not True or optimization.get("frozen_host_encoder_and_predictor") \
            is not True:
        raise SageMemSpecError("fair optimization contract changed")

    reporting = _mapping(spec.get("fairness_reporting"),
                         "fairness_reporting")
    if reporting.get("target_parameter_count_source") != \
            "exact per-cohort target_parameters field" \
            or reporting.get("exact_target_values") != {
                "SIGReg-LeWM": 76_032, "DINO-WM": 299_520} \
            or reporting.get("maximum_parameter_relative_gap") != 0.05 \
            or reporting.get("maximum_flop_relative_gap") != 0.10:
        raise SageMemSpecError("capacity fairness margins changed")
    required_fields = (
        "trainable_parameters", "forward_flops_per_episode",
        "persistent_state_floats", "peak_cuda_bytes",
        "wall_clock_train_seconds",
    )
    _exact_list(reporting.get("required_per_arm_fields"), required_fields,
                "required_per_arm_fields")

    statistics = _mapping(spec.get("statistics"), "statistics")
    if statistics.get("bootstrap_draws") != 20_000 \
            or statistics.get("resampling_unit") != \
            "formal seed and native episode cluster" \
            or statistics.get("preserve_pairing") != \
            "arms, reset interventions, context references, and counterfactual variants within episode":
        raise SageMemSpecError("paired-bootstrap contract changed")
    freshness = _mapping(spec.get("freshness"), "freshness")
    if freshness.get("development_source") != \
            "deterministic subset of parent TRAIN partition only" \
            or freshness.get("development_must_not_read_parent_validation_or_test") \
            is not True \
            or freshness.get("formal_preparation_status") != \
            "pending-executable-fresh-bank-builders" \
            or freshness.get(
                "formal_fail_closed_until_each_adapter_proves_parent-disjoint-selection") \
            is not True:
        raise SageMemSpecError("development/formal data boundary changed")
    gates = _mapping(spec.get("confirmatory_gates"), "confirmatory_gates")
    if _mapping(gates.get("next_feature_noninferiority"),
                "next_feature_noninferiority").get("relative_margin") != 0.02 \
            or _mapping(gates.get("reset_causality"),
                        "reset_causality").get(
                            "reset_to_full_mse_ratio_max") != 1.25:
        raise SageMemSpecError("MSE health targets changed")
    selection = _mapping(spec.get("development_selection"),
                         "development_selection")
    comparator_arms = (
        "gru", "lstm", "ssm", "fixed_trust", "gdelta",
        "fixed_trust_aux", "ssm_aux",
    )
    tie_break = (
        "ssm_aux", "fixed_trust_aux", "gdelta", "ssm", "fixed_trust",
        "gru", "lstm",
    )
    _exact_list(selection.get("comparator_arms"), comparator_arms,
                "development comparator arms")
    _exact_list(selection.get("tie_break_order"), tie_break,
                "development tie-break order")
    if selection.get("metrics") != {
            "retention": "maximize", "next_feature": "minimize",
            "execution": (
                "reuse-retention-comparator-without-development-execution-"
                "selection")} or not all(
                selection.get(key) is True for key in (
                    "requires_all_registered_cells",
                    "labels_allowed_only_for_posthoc_retention_and_execution_metrics",
                    "one_selection_receipt_per_cohort",
                    "global_audit_receipt_required_before_seal")):
        raise SageMemSpecError("development selection contract changed")
    gdelta_gate = _mapping(
        _mapping(spec.get("admission"), "admission").get(
            "gdelta_development_health_gate"), "gdelta health gate")
    if gdelta_gate != {
            "required": True, "next_feature_mse_ratio_max": 1.02,
            "reference_arms": [
                "gru", "lstm", "ssm", "fixed_trust", "fixed_trust_aux",
                "ssm_aux"],
            "finite_gradients_required": True,
            "exact_host_parameter_target_or_within_registered_5_percent":
                True}:
        raise SageMemSpecError("gDelta development health gate changed")
    execution = _mapping(spec.get("execution"), "execution")
    _exact_list(execution.get("allowed_physical_gpus"), (0, 1, 2),
                "allowed_physical_gpus")
    _exact_list(execution.get("stages"), STAGES, "stages")
    if execution.get("formal_confirmation_phrase") != \
            "RUN_SAGE_MEM_V1_FORMAL" or not all(
                execution.get(key) is True for key in (
                    "explicit_execute_required", "atomic_cell_directories",
                    "resume_requires_valid_manifest",
                    "never_overwrite_completed_cell",
                    "fail_closed_on_unexpected_files_or_status")):
        raise SageMemSpecError("execution safety contract changed")
    ownership = _mapping(execution.get("gpu_ownership"), "gpu_ownership")
    normalized = {int(key): tuple(value) for key, value in ownership.items()}
    if normalized != GPU_OWNERSHIP:
        raise SageMemSpecError("GPU ownership map changed")
    seed_registry(spec)


def load_spec(path: str | Path = DEFAULT_SPEC, *,
              verify_parent_paths: bool = False) -> dict[str, Any]:
    path = Path(path)
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise SageMemSpecError("spec root must be a mapping")
    validate_spec(value, verify_parent_paths=verify_parent_paths)
    value["_spec_path"] = str(path.resolve())
    value["_spec_sha256"] = sha256_file(path)
    value["_seed_registry"] = seed_registry(value)
    return value


def formal_cells(spec: Mapping[str, Any]) -> tuple[tuple[str, str, int], ...]:
    validate_spec(spec, verify_parent_paths=False)
    return tuple((cohort, arm, seed) for cohort in COHORTS for arm in ARMS
                 for seed in FORMAL_SEEDS)


def development_cells(spec: Mapping[str, Any]
                      ) -> tuple[tuple[str, str, int], ...]:
    validate_spec(spec, verify_parent_paths=False)
    return tuple((cohort, arm, seed) for cohort in COHORTS
                 for arm in DEVELOPMENT_ARMS for seed in DEVELOPMENT_SEEDS)


def output_root(spec: Mapping[str, Any]) -> Path:
    return resolve_repo_path(_mapping(spec.get("execution"), "execution")[
        "output_root"])


def cell_directory(spec: Mapping[str, Any], cohort: str, arm: str,
                   seed: int) -> Path:
    if (cohort, arm, seed) not in set(formal_cells(spec)):
        raise SageMemSpecError(f"unregistered formal cell: {cohort}/{arm}/{seed}")
    return output_root(spec) / "cells" / cohort / arm / f"seed-{seed}"


def development_cell_directory(spec: Mapping[str, Any], cohort: str,
                               arm: str, seed: int) -> Path:
    if (cohort, arm, seed) not in set(development_cells(spec)):
        raise SageMemSpecError(
            f"unregistered development cell: {cohort}/{arm}/{seed}")
    return (output_root(spec) / "development" / "cells" / cohort / arm
            / f"seed-{seed}")


def spec_fingerprint(spec: Mapping[str, Any]) -> str:
    value = {key: item for key, item in spec.items() if not key.startswith("_")}
    return sha256_bytes(canonical_json(value).encode("utf-8"))
