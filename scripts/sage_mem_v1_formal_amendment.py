#!/usr/bin/env python3
"""Validate the implementation-only SAGE-Mem formal amendment."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AMENDMENT = ROOT / "configs/sage_mem_v1_formal_amendment.yaml"
FORMAL_AMENDMENT_ID = "formal-two-phase-implementation-v1"


class FormalAmendmentError(ValueError):
    """The pre-formal implementation lock is absent or changed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise FormalAmendmentError(f"{label} must be a mapping")
    return value


def load_formal_amendment(
        path: str | Path = DEFAULT_AMENDMENT) -> dict[str, Any]:
    path = Path(path)
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise FormalAmendmentError("formal amendment root must be a mapping")
    if value.get("schema_version") != 1 \
            or value.get("study") != "sage-mem-v1" \
            or value.get("amendment") != FORMAL_AMENDMENT_ID \
            or value.get("status") != \
            "locked-before-development-selection-or-formal-data" \
            or value.get("scope") != \
            "implementation-only completion of preregistered pending formal hooks":
        raise FormalAmendmentError("formal amendment identity/scope changed")
    base = _mapping(value.get("base_protocol"), "base_protocol")
    if base.get("path") != "configs/sage_mem_v1.yaml":
        raise FormalAmendmentError("formal amendment base path changed")
    base_path = ROOT / str(base["path"])
    if not base_path.is_file() or _sha256(base_path) != base.get("sha256"):
        raise FormalAmendmentError("base protocol changed after amendment")
    fairness = _mapping(
        value.get("fairness_correction"), "fairness_correction")
    if fairness != {
        "trigger": (
            "deterministic parameter/FLOP ledger preflight before any "
            "complete development selection or formal run"),
        "invalid_partial_grid_archived": True,
        "outcome_dependent_choice": False,
        "thresholds_or_margins_changed": False,
        "candidate_revision": "two-dense-plus-diagonal-read-v1.1",
        "candidate_parameter_formula": "D(2D+A+2)",
        "gdelta_state_dim": {"SIGReg-LeWM": 95, "DINO-WM": 191},
        "rationale": (
            "replace a twice-applied dense read with a diagonal read plus "
            "one surprise projection; choose gDelta width nearest the "
            "unchanged target"),
    }:
        raise FormalAmendmentError("fairness correction changed")
    independence = _mapping(
        value.get("outcome_independence"), "outcome_independence")
    false_fields = {
        "development_selection_read", "aggregate_development_metrics_read",
        "formal_features_or_labels_read", "thresholds_changed",
        "cohorts_arms_ages_seeds_or_counts_changed",
        "optimization_or_objective_changed",
    }
    if set(independence) != false_fields \
            or any(independence[key] is not False for key in false_fields):
        raise FormalAmendmentError("outcome-independence declaration changed")
    phase_a = _mapping(value.get("phase_a"), "phase_a")
    phase_b = _mapping(value.get("phase_b"), "phase_b")
    endpoints = _mapping(value.get("causal_endpoints"), "causal_endpoints")
    execution = _mapping(value.get("execution"), "execution")
    if phase_a.get("cells") != 600 \
            or phase_a.get("formal_test_labels_available_to_workers") is not False \
            or phase_a.get("carrier_training_labels_forbidden") is not True \
            or phase_a.get("exact_bank_hash_required") is not True:
        raise FormalAmendmentError("Phase-A boundary changed")
    if phase_b.get("begins_only_after_complete_phase_a_validation") is not True \
            or phase_b.get("durable_label_reveal_receipt_required") is not True \
            or phase_b.get("evidence_ages_reported_separately") != [4, 8, 15] \
            or phase_b.get("pointmaze_counterfactuals_per_native_cluster") != 4:
        raise FormalAmendmentError("Phase-B boundary changed")
    raw_context = _mapping(
        phase_b.get("raw_context_reference"), "raw_context_reference")
    if raw_context != {
        "short_frames": 3,
        "long_frames": 16,
        "separate_from_parameter_matched_grid": True,
        "phase_a": "label-free equal-width frozen features only",
        "post_reveal_consumer": (
            "one shared short-long arm-blind LSQR readout per cohort and seed"),
        "mse_endpoint_registered": False,
    }:
        raise FormalAmendmentError("raw-context phase boundary changed")
    resets = _mapping(endpoints.get("reset_indices"), "reset_indices")
    if endpoints.get("frozen_host_output") != "primary" \
            or endpoints.get("carrier_prior") != "diagnostic" \
            or endpoints.get("reset_timing") != "immediately after cue offset" \
            or resets != {
                "lewm": {"age-4": 15, "age-8": 11, "age-15": 4},
                "dinowm": {"age-4": 4, "age-8": 4, "age-15": 4},
            }:
        raise FormalAmendmentError("causal endpoint definition changed")
    if execution != {
        "sealed_controller_decks_optional_per_cohort": True,
        "skipped_receipt_required_when_oracle_gate_fails": True,
        "program_level_use_claim_minimum_eligible_cohorts": 2,
        "pre_reveal_artifact": (
            "row x selected-class x true-target-class success cube"),
        "semantic_target_indexing": (
            "only after durable post-grid label reveal"),
        "unavailable_cohort_receipt_required": True,
        "program_claim_reported_separately_by_age": True,
    }:
        raise FormalAmendmentError("execution claim boundary changed")
    if value.get("claim_boundary") != \
            "no formal or paper claim is permitted from Phase-A artifacts alone":
        raise FormalAmendmentError("formal claim boundary changed")
    value["_path"] = str(path.resolve())
    value["_sha256"] = _sha256(path)
    return value


__all__ = [
    "DEFAULT_AMENDMENT", "FORMAL_AMENDMENT_ID", "FormalAmendmentError",
    "load_formal_amendment",
]
