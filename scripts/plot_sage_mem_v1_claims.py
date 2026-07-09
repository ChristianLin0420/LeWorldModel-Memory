#!/usr/bin/env python3
"""Render source-bound SAGE-Mem v1 claim figures from the paper ledger.

The only scientific input accepted by this renderer is the canonical claim
ledger emitted by ``scripts/summarize_sage_mem_v1_report.py``.  The renderer
does not open campaign cells, labels, predictions, checkpoints, or the formal
audit report.  It requires the caller to pin the ledger SHA-256, validates the
complete five-cohort by three-age grid, and refuses selective rows.

Execution writes six deterministic publication artifacts:

* a claim-ladder heatmap as vector PDF and 300-dpi PNG;
* an all-row interval plot as vector PDF and 300-dpi PNG;
* canonical plot data containing every registered row; and
* a canonical manifest binding the ledger, renderer, data, and figures.

Existing artifacts are never overwritten.  ``--resume`` only validates
byte-identical files and atomically repairs missing members of the same set.
Execution is confined to a symlink-free output directory below ``paper_a``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
LEDGER_SCHEMA = "sage_mem_v1_paper_claim_ledger_v1"
PLOT_DATA_SCHEMA = "sage_mem_v1_plot_data_v1"
PLOT_MANIFEST_SCHEMA = "sage_mem_v1_plot_manifest_v1"
REPORT_SCHEMA = "sage_mem_v1_formal_evidence_audit_v1"
DEFAULT_LEDGER = (
    ROOT / "paper_a/generated_results/sage_mem_v1_claim_ledger.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "paper_a/generated_results"
DEFAULT_PREFIX = "sage_mem_v1"

COHORTS = (
    "lewm_reacher_color",
    "lewm_pusht_color",
    "dinowm_pusht_token",
    "dinowm_pusht_binding",
    "dinowm_pointmaze_goal",
)
AGES = (4, 8, 15)
COHORT_LABELS = {
    "lewm_reacher_color": "LeWM Reacher color",
    "lewm_pusht_color": "LeWM PushT color",
    "dinowm_pusht_token": "DINO-WM PushT token",
    "dinowm_pusht_binding": "DINO-WM PushT binding",
    "dinowm_pointmaze_goal": "DINO-WM PointMaze goal",
}
GATE_KEYS = (
    "host_vs_locked_comparator",
    "full_vs_reset",
    "full_vs_none",
    "all_mechanism_controls",
    "next_mse_noninferiority",
)
CONTROL_KEYS = (
    "sage_mem_next_only",
    "sage_mem_no_exposure",
    "sage_mem_exposure_only",
    "fixed_trust_aux",
    "ssm_aux",
)
COMPARATOR_KEYS = ("retention", "next_feature", "execution")
BASELINE_ARMS = {
    "gru", "lstm", "ssm", "fixed_trust", "gdelta",
    "fixed_trust_aux", "ssm_aux",
}

SHA256_RE = re.compile(r"[0-9a-f]{64}")
PREFIX_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")

TOP_LEVEL_KEYS = {
    "schema", "study", "stage", "status", "integrity_completion",
    "claim_policy", "scientific_result", "cohort_summaries",
    "claim_rows", "execution_program", "source_binding",
    "publication_artifacts",
}
ROW_KEYS = {
    "cohort", "cohort_label", "age", "primary_endpoint",
    "host_full_accuracy", "host_full_vs_locked_comparator",
    "host_full_vs_reset", "host_full_vs_none",
    "reset_to_full_mse_ratio", "mechanism_controls",
    "next_feature_relative_excess", "gates", "primary_host_claim_pass",
    "prior_diagnostic", "raw_context_reference", "execution",
    "locked_comparators",
}
BOOTSTRAP_KEYS = {
    "point", "lower", "upper", "confidence", "draws", "seed",
    "resampling_unit", "class_profile_stratified", "pairing_preserved",
}
EXECUTION_KEYS = {
    "full_vs_locked_comparator", "full_vs_none", "full_vs_random",
    "random_reference", "random_reference_is_cohort_rate",
    "oracle_success", "random_success_mean", "pass",
}

# Okabe--Ito derived status colors; shapes/text preserve meaning in grayscale.
PASS = "#0072B2"
FAIL = "#D55E00"
NA = "#A7ADB2"
INK = "#202428"
MID = "#626970"
GRID = "#D9DDE0"
PALE = "#F5F6F7"
CHANNEL_COLORS = {
    "primary_host": "#0072B2",
    "prior_diagnostic": "#E69F00",
    "raw_context": "#009E73",
    "execution": "#CC79A7",
}
CHANNEL_MARKERS = {
    "primary_host": "o",
    "prior_diagnostic": "^",
    "raw_context": "s",
    "execution": "D",
}

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 7.0,
    "axes.titlesize": 7.7,
    "axes.labelsize": 7.0,
    "xtick.labelsize": 6.3,
    "ytick.labelsize": 6.3,
    "legend.fontsize": 6.2,
    "axes.edgecolor": MID,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MID,
    "ytick.color": MID,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.transparent": False,
})


class SageMemPlotError(RuntimeError):
    """The authenticated claim ledger cannot be safely plotted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SageMemPlotError(message)


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ) + "\n").encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_stable(path: Path, label: str) -> bytes:
    _require(path.is_file() and not path.is_symlink(),
             f"{label} is missing or unsafe: {path}")
    before = path.stat()
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise SageMemPlotError(f"cannot read {label}: {path}") from error
    after = path.stat()
    identity_before = (
        before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    _require(identity_before == identity_after,
             f"{label} changed while being read: {path}")
    return payload


def _decode_canonical(payload: bytes, label: str) -> dict[str, Any]:
    def reject_duplicates(
            pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite constant: {token}")),
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise SageMemPlotError(f"{label} is not strict UTF-8 JSON") from error
    _require(isinstance(value, dict), f"{label} root must be a mapping")
    _require(payload == _canonical_json(value),
             f"{label} is not canonical JSON")
    return value


def _mapping(value: Any, label: str,
             keys: set[str] | None = None) -> Mapping[str, Any]:
    _require(isinstance(value, dict), f"{label} must be a mapping")
    if keys is not None:
        _require(set(value) == keys,
                 f"{label} keys changed: expected {sorted(keys)}, "
                 f"observed {sorted(value)}")
    return value


def _list(value: Any, label: str) -> list[Any]:
    _require(isinstance(value, list), f"{label} must be a list")
    return value


def _bool(value: Any, label: str) -> bool:
    _require(isinstance(value, bool), f"{label} must be boolean")
    return value


def _integer(value: Any, label: str, minimum: int | None = None) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool),
             f"{label} must be an integer")
    if minimum is not None:
        _require(value >= minimum, f"{label} must be at least {minimum}")
    return value


def _number(value: Any, label: str, *, lower: float | None = None,
            upper: float | None = None) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool),
             f"{label} must be numeric")
    result = float(value)
    _require(math.isfinite(result), f"{label} must be finite")
    if lower is not None:
        _require(result >= lower, f"{label} is below {lower}")
    if upper is not None:
        _require(result <= upper, f"{label} is above {upper}")
    return result


def _string(value: Any, label: str) -> str:
    _require(isinstance(value, str) and bool(value),
             f"{label} must be a nonempty string")
    return value


def _digest(value: Any, label: str) -> str:
    _require(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
             f"{label} must be a lowercase SHA-256 digest")
    return value


def _exact_number(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)


def _validate_bootstrap(value: Any, label: str, *, confidence: float,
                        lower_bound: float | None = None,
                        upper_bound: float | None = None
                        ) -> dict[str, Any]:
    record = _mapping(value, label, BOOTSTRAP_KEYS)
    point = _number(record["point"], f"{label}.point",
                    lower=lower_bound, upper=upper_bound)
    lower = _number(record["lower"], f"{label}.lower",
                    lower=lower_bound, upper=upper_bound)
    upper = _number(record["upper"], f"{label}.upper",
                    lower=lower_bound, upper=upper_bound)
    _require(lower <= point <= upper,
             f"{label} point is outside its confidence interval")
    observed_confidence = _number(
        record["confidence"], f"{label}.confidence", lower=0.0, upper=1.0)
    _require(_exact_number(observed_confidence, confidence),
             f"{label} confidence must be {confidence}")
    draws = _integer(record["draws"], f"{label}.draws", minimum=1)
    seed = _integer(record["seed"], f"{label}.seed", minimum=0)
    resampling = _string(record["resampling_unit"],
                         f"{label}.resampling_unit")
    _require(record["class_profile_stratified"] is True
             and record["pairing_preserved"] is True,
             f"{label} lost registered pairing/stratification")
    return {
        "point": point,
        "lower": lower,
        "upper": upper,
        "confidence": observed_confidence,
        "draws": draws,
        "seed": seed,
        "resampling_unit": resampling,
        "class_profile_stratified": True,
        "pairing_preserved": True,
    }


def _validate_identity(value: Any, label: str, *, status: str | None = None
                       ) -> dict[str, Any]:
    record = _mapping(value, label, {"path", "size", "sha256", "status"})
    result = {
        "path": _string(record["path"], f"{label}.path"),
        "size": _integer(record["size"], f"{label}.size", minimum=1),
        "sha256": _digest(record["sha256"], f"{label}.sha256"),
        "status": _string(record["status"], f"{label}.status"),
    }
    if status is not None:
        _require(result["status"] == status,
                 f"{label} status must be {status}")
    return result


def _validate_source_binding(value: Any) -> dict[str, Any]:
    source = _mapping(value, "source_binding", {
        "report", "protocol", "formal_auditor", "adapter",
    })
    report = _mapping(source["report"], "source_binding.report", {
        "path", "size", "sha256", "schema", "expected_sha256_verified",
        "independent_sealed_auditor_recomputation_verified",
        "standard_roots",
    })
    roots = _mapping(
        report["standard_roots"], "source_binding.report.standard_roots",
        {"phase_a", "finalized", "preparation", "raw_context"})
    _require(report["schema"] == REPORT_SCHEMA
             and report["expected_sha256_verified"] is True
             and report[
                 "independent_sealed_auditor_recomputation_verified"] is True,
             "formal-report source authentication is incomplete")
    report_normalized = {
        "path": _string(report["path"], "source_binding.report.path"),
        "size": _integer(report["size"], "source_binding.report.size", 1),
        "sha256": _digest(report["sha256"],
                          "source_binding.report.sha256"),
        "schema": REPORT_SCHEMA,
        "expected_sha256_verified": True,
        "independent_sealed_auditor_recomputation_verified": True,
        "standard_roots": {
            key: _string(roots[key],
                         f"source_binding.report.standard_roots.{key}")
            for key in ("phase_a", "finalized", "preparation", "raw_context")
        },
    }

    protocol = _mapping(source["protocol"], "source_binding.protocol", {
        "path", "size", "sha256", "fingerprint", "implementation_lock",
        "formal_amendment", "report_schema_repeats_protocol_fingerprint",
        "binding_note",
    })
    _require(protocol["report_schema_repeats_protocol_fingerprint"] is False,
             "protocol/report binding boundary changed")
    lock = _validate_identity(
        protocol["implementation_lock"],
        "source_binding.protocol.implementation_lock", status="sealed")
    amendment = _validate_identity(
        protocol["formal_amendment"],
        "source_binding.protocol.formal_amendment",
        status="locked-before-development-selection-or-formal-data")
    protocol_normalized = {
        "path": _string(protocol["path"], "source_binding.protocol.path"),
        "size": _integer(protocol["size"],
                         "source_binding.protocol.size", 1),
        "sha256": _digest(protocol["sha256"],
                          "source_binding.protocol.sha256"),
        "fingerprint": _digest(
            protocol["fingerprint"], "source_binding.protocol.fingerprint"),
        "implementation_lock": lock,
        "formal_amendment": amendment,
        "report_schema_repeats_protocol_fingerprint": False,
        "binding_note": _string(
            protocol["binding_note"], "source_binding.protocol.binding_note"),
    }

    auditor = _mapping(source["formal_auditor"],
                       "source_binding.formal_auditor", {
                           "path", "size", "sha256",
                           "sealed_by_implementation_lock",
                       })
    _require(auditor["sealed_by_implementation_lock"] is True,
             "formal auditor is not sealed by the implementation lock")
    auditor_normalized = {
        "path": _string(auditor["path"],
                        "source_binding.formal_auditor.path"),
        "size": _integer(auditor["size"],
                         "source_binding.formal_auditor.size", 1),
        "sha256": _digest(auditor["sha256"],
                          "source_binding.formal_auditor.sha256"),
        "sealed_by_implementation_lock": True,
    }
    adapter = _mapping(source["adapter"], "source_binding.adapter",
                       {"path", "sha256"})
    adapter_path = _string(adapter["path"], "source_binding.adapter.path")
    _require(adapter_path.replace("\\", "/").endswith(
        "scripts/summarize_sage_mem_v1_report.py"),
        "source adapter path is not the registered report adapter")
    adapter_normalized = {
        "path": adapter_path,
        "sha256": _digest(adapter["sha256"],
                          "source_binding.adapter.sha256"),
    }
    return {
        "report": report_normalized,
        "protocol": protocol_normalized,
        "formal_auditor": auditor_normalized,
        "adapter": adapter_normalized,
    }


def _validate_row(value: Any, *, cohort: str, age: int,
                  thresholds: Mapping[str, float]) -> dict[str, Any]:
    label = f"claim_rows.{cohort}.age_{age}"
    row = _mapping(value, label, ROW_KEYS)
    _require(row["cohort"] == cohort
             and row["cohort_label"] == COHORT_LABELS[cohort]
             and row["age"] == age,
             f"{label} identity/order changed")
    _require(row["primary_endpoint"] == "frozen-host full correctness",
             f"{label} primary endpoint changed")

    host_accuracy = _validate_bootstrap(
        row["host_full_accuracy"], f"{label}.host_full_accuracy",
        confidence=0.95, lower_bound=0.0, upper_bound=1.0)
    host = _validate_bootstrap(
        row["host_full_vs_locked_comparator"],
        f"{label}.host_full_vs_locked_comparator", confidence=0.95,
        lower_bound=-1.0, upper_bound=1.0)
    reset = _validate_bootstrap(
        row["host_full_vs_reset"], f"{label}.host_full_vs_reset",
        confidence=0.95, lower_bound=-1.0, upper_bound=1.0)
    none = _validate_bootstrap(
        row["host_full_vs_none"], f"{label}.host_full_vs_none",
        confidence=0.95, lower_bound=-1.0, upper_bound=1.0)
    ratio = _number(row["reset_to_full_mse_ratio"],
                    f"{label}.reset_to_full_mse_ratio", lower=0.0)
    controls_value = _mapping(
        row["mechanism_controls"], f"{label}.mechanism_controls",
        set(CONTROL_KEYS))
    controls = {
        key: _validate_bootstrap(
            controls_value[key], f"{label}.mechanism_controls.{key}",
            confidence=0.95, lower_bound=-1.0, upper_bound=1.0)
        for key in CONTROL_KEYS
    }
    next_mse = _validate_bootstrap(
        row["next_feature_relative_excess"],
        f"{label}.next_feature_relative_excess", confidence=0.90,
        lower_bound=-1.0)

    expected_gates = {
        "host_vs_locked_comparator":
            host["lower"] >= thresholds["host_gain"],
        "full_vs_reset": (
            reset["lower"] >= thresholds["reset_gain"]
            and ratio <= thresholds["reset_mse_ratio_max"]),
        "full_vs_none": none["lower"] >= thresholds["mechanism_gain"],
        "all_mechanism_controls": all(
            item["lower"] >= thresholds["mechanism_gain"]
            for item in controls.values()),
        "next_mse_noninferiority":
            next_mse["upper"] <= thresholds["mse_relative_margin"],
    }
    gates_value = _mapping(row["gates"], f"{label}.gates", set(GATE_KEYS))
    gates = {
        key: _bool(gates_value[key], f"{label}.gates.{key}")
        for key in GATE_KEYS
    }
    _require(gates == expected_gates,
             f"{label} gate flags differ from registered intervals")
    primary_pass = all(expected_gates.values())
    _require(_bool(row["primary_host_claim_pass"],
                   f"{label}.primary_host_claim_pass") == primary_pass,
             f"{label} primary conjunction is inconsistent")

    prior_value = _mapping(row["prior_diagnostic"],
                           f"{label}.prior_diagnostic", {
                               "role", "accuracy", "vs_locked_comparator",
                               "resolved_positive", "enters_primary_host_claim",
                           })
    _require(prior_value["role"] ==
             "diagnostic-only; cannot establish host use"
             and prior_value["enters_primary_host_claim"] is False,
             f"{label} prior crossed the primary-claim boundary")
    prior_accuracy = _validate_bootstrap(
        prior_value["accuracy"], f"{label}.prior_diagnostic.accuracy",
        confidence=0.95, lower_bound=0.0, upper_bound=1.0)
    prior_effect = _validate_bootstrap(
        prior_value["vs_locked_comparator"],
        f"{label}.prior_diagnostic.vs_locked_comparator", confidence=0.95,
        lower_bound=-1.0, upper_bound=1.0)
    prior_positive = prior_effect["lower"] > 0.0
    _require(_bool(prior_value["resolved_positive"],
                   f"{label}.prior_diagnostic.resolved_positive")
             == prior_positive,
             f"{label} prior resolution is inconsistent")

    raw_value = _mapping(row["raw_context_reference"],
                         f"{label}.raw_context_reference", {
                             "short3_accuracy", "long16_accuracy",
                             "long16_minus_short3",
                             "resolved_long_context_gain",
                             "separate_from_parameter_matched_grid",
                         })
    short_accuracy = _number(
        raw_value["short3_accuracy"],
        f"{label}.raw_context_reference.short3_accuracy",
        lower=0.0, upper=1.0)
    long_accuracy = _number(
        raw_value["long16_accuracy"],
        f"{label}.raw_context_reference.long16_accuracy",
        lower=0.0, upper=1.0)
    raw_effect = _validate_bootstrap(
        raw_value["long16_minus_short3"],
        f"{label}.raw_context_reference.long16_minus_short3",
        confidence=0.95, lower_bound=-1.0, upper_bound=1.0)
    raw_positive = raw_effect["lower"] > 0.0
    _require(raw_value["separate_from_parameter_matched_grid"] is True
             and _bool(raw_value["resolved_long_context_gain"],
                       f"{label}.raw_context_reference.resolved")
             == raw_positive,
             f"{label} raw-context resolution is inconsistent")

    execution = None
    if row["execution"] is not None:
        execution_value = _mapping(
            row["execution"], f"{label}.execution", EXECUTION_KEYS)
        contrasts = {
            key: _validate_bootstrap(
                execution_value[key], f"{label}.execution.{key}",
                confidence=0.95, lower_bound=-1.0, upper_bound=1.0)
            for key in (
                "full_vs_locked_comparator", "full_vs_none",
                "full_vs_random")
        }
        _require(execution_value["random_reference"] ==
                 "sealed per-episode arm-blind random-success deck"
                 and execution_value[
                     "random_reference_is_cohort_rate"] is False,
                 f"{label} execution random-reference contract changed")
        oracle = _number(execution_value["oracle_success"],
                         f"{label}.execution.oracle_success",
                         lower=0.0, upper=1.0)
        random_mean = _number(execution_value["random_success_mean"],
                              f"{label}.execution.random_success_mean",
                              lower=0.0, upper=1.0)
        _require(oracle >= thresholds["execution_oracle_gate"],
                 f"{label} execution bypasses the oracle gate")
        execution_pass = all(
            contrast["lower"] >= thresholds["execution_gain"]
            for contrast in contrasts.values())
        _require(_bool(execution_value["pass"],
                       f"{label}.execution.pass") == execution_pass,
                 f"{label} execution conjunction is inconsistent")
        execution = {
            **contrasts,
            "random_reference": execution_value["random_reference"],
            "random_reference_is_cohort_rate": False,
            "oracle_success": oracle,
            "random_success_mean": random_mean,
            "pass": execution_pass,
        }

    comparators_value = _mapping(
        row["locked_comparators"], f"{label}.locked_comparators",
        set(COMPARATOR_KEYS))
    comparators = {
        key: _string(comparators_value[key],
                     f"{label}.locked_comparators.{key}")
        for key in COMPARATOR_KEYS
    }
    _require(all(item in BASELINE_ARMS for item in comparators.values()),
             f"{label} contains an invalid locked comparator")
    return {
        "cohort": cohort,
        "cohort_label": COHORT_LABELS[cohort],
        "age": age,
        "primary_endpoint": "frozen-host full correctness",
        "host_full_accuracy": host_accuracy,
        "host_full_vs_locked_comparator": host,
        "host_full_vs_reset": reset,
        "host_full_vs_none": none,
        "reset_to_full_mse_ratio": ratio,
        "mechanism_controls": controls,
        "next_feature_relative_excess": next_mse,
        "gates": gates,
        "primary_host_claim_pass": primary_pass,
        "prior_diagnostic": {
            "role": prior_value["role"],
            "accuracy": prior_accuracy,
            "vs_locked_comparator": prior_effect,
            "resolved_positive": prior_positive,
            "enters_primary_host_claim": False,
        },
        "raw_context_reference": {
            "short3_accuracy": short_accuracy,
            "long16_accuracy": long_accuracy,
            "long16_minus_short3": raw_effect,
            "resolved_long_context_gain": raw_positive,
            "separate_from_parameter_matched_grid": True,
        },
        "execution": execution,
        "locked_comparators": comparators,
    }


def _validate_cohort_summaries(value: Any,
                               rows: Sequence[Mapping[str, Any]]) -> None:
    summaries = _mapping(value, "cohort_summaries")
    _require(set(summaries) == set(COHORTS),
             "cohort summaries do not contain exactly five cohorts")
    summary_keys = {
        "cohort_label", "locked_comparators", "comparator_receipt",
        "backend_admission", "resource_enforcement_verified",
        "registered_age_rows", "rows_passing_primary_host_claim",
        "all_registered_ages_primary_host_claim_pass", "execution_supplied",
        "execution_pass_by_age",
    }
    receipt_keys = {
        "formal_preparation_manifest_sha256", "implementation_lock_sha256",
        "custody_registry_sha256", "preparation_receipt",
        "locked_comparator_receipt",
    }
    for cohort in COHORTS:
        label = f"cohort_summaries.{cohort}"
        summary = _mapping(summaries[cohort], label, summary_keys)
        cohort_rows = [row for row in rows if row["cohort"] == cohort]
        _require(summary["cohort_label"] == COHORT_LABELS[cohort]
                 and summary["registered_age_rows"] == 3,
                 f"{label} identity/count changed")
        comparators = _mapping(
            summary["locked_comparators"], f"{label}.locked_comparators",
            set(COMPARATOR_KEYS))
        _require(dict(comparators) == cohort_rows[0]["locked_comparators"]
                 and all(row["locked_comparators"] == dict(comparators)
                         for row in cohort_rows),
                 f"{label} comparators differ from claim rows")
        receipt = _mapping(summary["comparator_receipt"],
                           f"{label}.comparator_receipt", receipt_keys)
        for key in (
                "formal_preparation_manifest_sha256",
                "implementation_lock_sha256", "custody_registry_sha256"):
            _digest(receipt[key], f"{label}.comparator_receipt.{key}")
        for key in ("preparation_receipt", "locked_comparator_receipt"):
            identity = _mapping(receipt[key],
                                f"{label}.comparator_receipt.{key}",
                                {"path", "size", "sha256"})
            _string(identity["path"], f"{label}.{key}.path")
            _integer(identity["size"], f"{label}.{key}.size", 1)
            _digest(identity["sha256"], f"{label}.{key}.sha256")
        _mapping(summary["backend_admission"],
                 f"{label}.backend_admission")
        _require(summary["resource_enforcement_verified"] is True,
                 f"{label} resources were not verified")
        passing = sum(row["primary_host_claim_pass"] for row in cohort_rows)
        _require(summary["rows_passing_primary_host_claim"] == passing
                 and summary[
                     "all_registered_ages_primary_host_claim_pass"]
                 is (passing == 3),
                 f"{label} primary summary differs from rows")
        execution_supplied = all(
            row["execution"] is not None for row in cohort_rows)
        _require(summary["execution_supplied"] is execution_supplied,
                 f"{label} execution presence differs from rows")
        pass_by_age = _mapping(
            summary["execution_pass_by_age"],
            f"{label}.execution_pass_by_age", {"4", "8", "15"})
        for row in cohort_rows:
            expected = (row["execution"]["pass"]
                        if row["execution"] is not None else None)
            _require(pass_by_age[str(row["age"])] is expected,
                     f"{label} execution age summary differs from rows")


def validate_ledger(value: Any) -> dict[str, Any]:
    """Strictly validate and normalize one claim ledger."""

    ledger = _mapping(value, "claim ledger", TOP_LEVEL_KEYS)
    _require(ledger["schema"] == LEDGER_SCHEMA
             and ledger["study"] == "sage-mem-v1"
             and ledger["stage"] == "paper-claim-ledger"
             and ledger["status"] == "complete",
             "claim-ledger identity/status changed or is incomplete")

    integrity = _mapping(ledger["integrity_completion"],
                         "integrity_completion", {
                             "status", "meaning", "phase_a_cells_verified",
                             "finalized_cells_verified", "comparators_verified",
                             "resources_verified",
                             "raw_context_references_verified",
                             "phase_a_grid_sha256", "identity_ledger_sha256",
                         })
    _require(integrity["status"] == "complete"
             and integrity["phase_a_cells_verified"] == 600
             and integrity["finalized_cells_verified"] == 600
             and integrity["comparators_verified"] == 5
             and integrity["resources_verified"] == 600
             and integrity["raw_context_references_verified"] == 50,
             "claim-ledger integrity counts are incomplete")
    _string(integrity["meaning"], "integrity_completion.meaning")
    _digest(integrity["phase_a_grid_sha256"],
            "integrity_completion.phase_a_grid_sha256")
    _digest(integrity["identity_ledger_sha256"],
            "integrity_completion.identity_ledger_sha256")

    policy = _mapping(ledger["claim_policy"], "claim_policy", {
        "per_age_claims_only", "registered_cohorts", "registered_ages",
        "registered_claim_rows", "positive_rows_may_not_be_selected_or_omitted",
        "prior_can_substitute_for_host_output",
        "pooled_cross_host_score_computed",
        "universal_success_claim_permitted", "thresholds",
    })
    _require(policy["per_age_claims_only"] is True
             and policy["registered_cohorts"] == list(COHORTS)
             and policy["registered_ages"] == list(AGES)
             and policy["registered_claim_rows"] == 15
             and policy[
                 "positive_rows_may_not_be_selected_or_omitted"] is True
             and policy["prior_can_substitute_for_host_output"] is False
             and policy["pooled_cross_host_score_computed"] is False
             and policy["universal_success_claim_permitted"] is False,
             "claim policy permits selection, pooling, or substitution")
    threshold_keys = {
        "host_gain", "reset_gain", "reset_mse_ratio_max",
        "mechanism_gain", "mse_relative_margin", "execution_gain",
        "execution_oracle_gate",
    }
    thresholds_value = _mapping(
        policy["thresholds"], "claim_policy.thresholds", threshold_keys)
    thresholds = {
        key: _number(thresholds_value[key], f"claim_policy.thresholds.{key}")
        for key in sorted(threshold_keys)
    }
    _require(thresholds["host_gain"] >= 0.0
             and thresholds["reset_gain"] >= 0.0
             and thresholds["mechanism_gain"] >= 0.0
             and thresholds["execution_gain"] >= 0.0
             and thresholds["reset_mse_ratio_max"] >= 0.0
             and 0.0 <= thresholds["execution_oracle_gate"] <= 1.0,
             "registered threshold domains changed")

    rows_value = _list(ledger["claim_rows"], "claim_rows")
    _require(len(rows_value) == 15,
             "claim ledger must contain all 15 registered rows")
    expected_order = [
        (cohort, age) for cohort in COHORTS for age in AGES
    ]
    observed_order = [
        (row.get("cohort"), row.get("age"))
        if isinstance(row, dict) else (None, None)
        for row in rows_value
    ]
    _require(observed_order == expected_order,
             "claim rows are missing, duplicated, selected, or reordered")
    rows = [
        _validate_row(row, cohort=cohort, age=age, thresholds=thresholds)
        for row, (cohort, age) in zip(
            rows_value, expected_order, strict=True)
    ]

    scientific = _mapping(ledger["scientific_result"],
                          "scientific_result", {
                              "status", "primary_claim_rows_total",
                              "primary_claim_rows_passing",
                              "primary_claim_rows_failing",
                              "any_primary_claim_row_passed",
                              "all_primary_claim_rows_passed", "meaning",
                          })
    passing = sum(row["primary_host_claim_pass"] for row in rows)
    _require(scientific["status"] == "evaluated"
             and scientific["primary_claim_rows_total"] == 15
             and scientific["primary_claim_rows_passing"] == passing
             and scientific["primary_claim_rows_failing"] == 15 - passing
             and scientific["any_primary_claim_row_passed"] is (passing > 0)
             and scientific["all_primary_claim_rows_passed"] is
             (passing == 15),
             "scientific-result counts differ from all registered rows")
    _string(scientific["meaning"], "scientific_result.meaning")
    _validate_cohort_summaries(ledger["cohort_summaries"], rows)

    program = _mapping(ledger["execution_program"], "execution_program", {
        "optional", "eligible_cohorts", "minimum_eligible_cohorts",
        "program_claim_permitted", "per_age",
        "cross_age_conjunction_computed", "program_claim_pass",
    })
    eligible_cohorts = sum(
        all(row["execution"] is not None
            for row in rows if row["cohort"] == cohort)
        for cohort in COHORTS)
    permitted = eligible_cohorts >= 2
    _require(program["optional"] is True
             and program["eligible_cohorts"] == eligible_cohorts
             and program["minimum_eligible_cohorts"] == 2
             and program["program_claim_permitted"] is permitted
             and program["cross_age_conjunction_computed"] is False
             and program["program_claim_pass"] is None,
             "optional execution-program boundary changed")
    per_age = _mapping(program["per_age"], "execution_program.per_age",
                       {"4", "8", "15"})
    for age in AGES:
        age_record = _mapping(
            per_age[str(age)], f"execution_program.per_age.{age}", {
                "eligible_cohorts", "cohorts_passing", "claim_permitted",
                "claim_pass",
            })
        age_passing = sum(
            row["execution"] is not None and row["execution"]["pass"]
            for row in rows if row["age"] == age)
        _require(age_record["eligible_cohorts"] == eligible_cohorts
                 and age_record["cohorts_passing"] == age_passing
                 and age_record["claim_permitted"] is permitted
                 and age_record["claim_pass"] is
                 (permitted and age_passing >= 2),
                 f"execution-program age-{age} summary differs from rows")

    source = _validate_source_binding(ledger["source_binding"])
    publication = _mapping(ledger["publication_artifacts"],
                           "publication_artifacts", {"tex"})
    tex = _mapping(publication["tex"], "publication_artifacts.tex",
                   {"path", "sha256"})
    publication_normalized = {
        "tex": {
            "path": _string(tex["path"], "publication_artifacts.tex.path"),
            "sha256": _digest(tex["sha256"],
                              "publication_artifacts.tex.sha256"),
        },
    }
    return {
        "schema": LEDGER_SCHEMA,
        "study": "sage-mem-v1",
        "stage": "paper-claim-ledger",
        "status": "complete",
        "integrity_completion": dict(integrity),
        "claim_policy": {
            **dict(policy),
            "thresholds": thresholds,
        },
        "scientific_result": dict(scientific),
        "cohort_summaries": dict(ledger["cohort_summaries"]),
        "claim_rows": rows,
        "execution_program": dict(program),
        "source_binding": source,
        "publication_artifacts": publication_normalized,
    }


def _status_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        execution = row["execution"]
        result.append({
            "cohort": row["cohort"],
            "cohort_label": row["cohort_label"],
            "age": row["age"],
            "primary_gates": dict(row["gates"]),
            "primary_host_claim": row["primary_host_claim_pass"],
            "prior_diagnostic": row["prior_diagnostic"]["resolved_positive"],
            "raw_context": row["raw_context_reference"][
                "resolved_long_context_gain"],
            "execution": execution["pass"] if execution is not None else None,
        })
    return result


def _plot_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Preserve all registered evidence while defining plotted contrasts."""

    result = []
    for row in rows:
        execution = row["execution"]
        result.append({
            "cohort": row["cohort"],
            "cohort_label": row["cohort_label"],
            "age": row["age"],
            "primary_host": {
                "host_full_vs_locked_comparator":
                    row["host_full_vs_locked_comparator"],
                "host_full_vs_reset": row["host_full_vs_reset"],
                "host_full_vs_none": row["host_full_vs_none"],
                "mechanism_controls": row["mechanism_controls"],
                "next_feature_relative_excess":
                    row["next_feature_relative_excess"],
                "reset_to_full_mse_ratio":
                    row["reset_to_full_mse_ratio"],
                "gates": row["gates"],
                "claim_pass": row["primary_host_claim_pass"],
            },
            "prior_diagnostic": {
                "vs_locked_comparator":
                    row["prior_diagnostic"]["vs_locked_comparator"],
                "resolved_positive":
                    row["prior_diagnostic"]["resolved_positive"],
                "enters_primary_host_claim": False,
            },
            "raw_context": {
                "long16_minus_short3":
                    row["raw_context_reference"]["long16_minus_short3"],
                "resolved_long_context_gain":
                    row["raw_context_reference"][
                        "resolved_long_context_gain"],
                "separate_from_parameter_matched_grid": True,
            },
            "execution": execution,
        })
    return result


def build_plot_data(ledger: Mapping[str, Any], *, ledger_sha256: str,
                    script_sha256: str) -> dict[str, Any]:
    rows = ledger["claim_rows"]
    return {
        "schema": PLOT_DATA_SCHEMA,
        "study": "sage-mem-v1",
        "stage": "publication-plot-data",
        "status": "complete",
        "source_binding": {
            "claim_ledger_sha256": ledger_sha256,
            "claim_ledger_schema": LEDGER_SCHEMA,
            "formal_report_sha256":
                ledger["source_binding"]["report"]["sha256"],
            "protocol_fingerprint":
                ledger["source_binding"]["protocol"]["fingerprint"],
            "report_adapter_sha256":
                ledger["source_binding"]["adapter"]["sha256"],
            "plotting_script_sha256": script_sha256,
        },
        "selection_policy": {
            "registered_rows_required": 15,
            "all_five_cohorts_and_all_three_ages_included": True,
            "row_order": [
                {"cohort": cohort, "age": age}
                for cohort in COHORTS for age in AGES
            ],
            "positive_rows_may_not_be_selected_or_omitted": True,
            "primary_host_plot_contrast":
                "host_full_vs_locked_comparator",
            "prior_plot_contrast": "vs_locked_comparator",
            "raw_context_plot_contrast": "long16_minus_short3",
            "execution_plot_contrast": "full_vs_locked_comparator",
            "optional_execution_missingness_is_explicit": True,
            "shared_symmetric_effect_axis_across_channels": True,
        },
        "thresholds": dict(ledger["claim_policy"]["thresholds"]),
        "claim_ladder_rows": _status_rows(rows),
        "effect_rows": _plot_rows(rows),
    }


def _row_labels(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [f"{row['cohort_label']}  |  {row['age']}" for row in rows]


def _figure_bytes(fig: plt.Figure, *, ledger_sha256: str,
                  script_sha256: str) -> tuple[bytes, bytes]:
    pdf = io.BytesIO()
    png = io.BytesIO()
    fig.savefig(
        pdf, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.035,
        metadata={
            "Title": "SAGE-Mem v1 registered claim evidence",
            "Author": "SAGE-Mem v1 audit",
            "Subject": f"claim-ledger-sha256={ledger_sha256}",
            "Keywords": f"plot-script-sha256={script_sha256}",
            "Creator": "plot_sage_mem_v1_claims.py",
            "Producer": "Matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(
        png, format="png", dpi=300, bbox_inches="tight", pad_inches=0.035,
        metadata={
            "Software": "plot_sage_mem_v1_claims.py",
            "ClaimLedgerSHA256": ledger_sha256,
            "PlotScriptSHA256": script_sha256,
        },
    )
    plt.close(fig)
    return pdf.getvalue(), png.getvalue()


def render_claim_ladder(plot_data: Mapping[str, Any], *, ledger_sha256: str,
                        script_sha256: str) -> tuple[bytes, bytes]:
    rows = plot_data["claim_ladder_rows"]
    columns = (
        ("Host >\nlocked", "primary_gates", "host_vs_locked_comparator"),
        ("Full >\nreset", "primary_gates", "full_vs_reset"),
        ("Full >\nno state", "primary_gates", "full_vs_none"),
        ("All mech.\ncontrols", "primary_gates", "all_mechanism_controls"),
        ("Next-fit\nnoninferior", "primary_gates",
         "next_mse_noninferiority"),
        ("Primary\nclaim", "primary_host_claim", None),
        ("Carrier\nprior", "prior_diagnostic", None),
        ("Long raw\ncontext", "raw_context", None),
        ("Executed\nuse", "execution", None),
    )
    statuses: list[list[bool | None]] = []
    for row in rows:
        row_status = []
        for _, group, key in columns:
            value = row[group]
            row_status.append(value[key] if key is not None else value)
        statuses.append(row_status)

    fig, ax = plt.subplots(figsize=(7.05, 4.05))
    ax.set_xlim(-0.5, len(columns) - 0.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_aspect("auto")
    for y, status_row in enumerate(statuses):
        for x, status in enumerate(status_row):
            color = PASS if status is True else FAIL if status is False else NA
            ax.add_patch(Rectangle(
                (x - 0.47, y - 0.43), 0.94, 0.86,
                facecolor=color, edgecolor="white", linewidth=0.65))
            if status is True:
                ax.scatter(x, y, s=19, facecolors="none", edgecolors="white",
                           linewidths=1.05, marker="o", zorder=3)
            elif status is False:
                ax.scatter(x, y, s=19, color="white", linewidths=1.05,
                           marker="x", zorder=3)
            else:
                ax.plot([x - 0.12, x + 0.12], [y, y], color="white",
                        linewidth=1.25, solid_capstyle="round", zorder=3)
    for boundary in (2.5, 5.5, 6.5, 7.5):
        ax.axvline(boundary, color=INK, linewidth=0.65, alpha=0.75)
    for boundary in (2.5, 5.5, 8.5, 11.5):
        ax.axhline(boundary, color=INK, linewidth=0.65, alpha=0.75)
    ax.set_xticks(range(len(columns)), [item[0] for item in columns])
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=3.0)
    ax.set_yticks(range(len(rows)), _row_labels(rows))
    ax.tick_params(axis="y", length=0, pad=3.5)
    for label in ax.get_yticklabels():
        label.set_horizontalalignment("right")
    for spine in ax.spines.values():
        spine.set_visible(False)

    group_spans = (
        (-0.48, 5.48, "PRIMARY FROZEN-HOST GATES",
         CHANNEL_COLORS["primary_host"]),
        (5.52, 6.48, "PRIOR", CHANNEL_COLORS["prior_diagnostic"]),
        (6.52, 7.48, "RAW CTX.", CHANNEL_COLORS["raw_context"]),
        (7.52, 8.48, "EXECUTE", CHANNEL_COLORS["execution"]),
    )
    for left, right, label, color in group_spans:
        width = right - left
        ax.add_patch(Rectangle(
            (left, 1.115), width, 0.075,
            transform=ax.get_xaxis_transform(), clip_on=False,
            facecolor=color, edgecolor="none"))
        ax.text((left + right) / 2, 1.152, label,
                transform=ax.get_xaxis_transform(), ha="center", va="center",
                fontsize=5.55, color="white", fontweight="bold",
                clip_on=False)

    legend = [
        Line2D([0], [0], marker="o", markersize=4.7, markerfacecolor=PASS,
               markeredgecolor="white", linewidth=0, label="Pass"),
        Line2D([0], [0], marker="x", markersize=4.7, color=FAIL,
               linewidth=0, label="Fail"),
        Patch(facecolor=NA, edgecolor="none", label="Not evaluated"),
    ]
    ax.legend(handles=legend, ncol=3, frameon=False, loc="lower left",
              bbox_to_anchor=(0.0, -0.145), handletextpad=0.45,
              columnspacing=1.25, borderaxespad=0.0)
    ax.text(
        1.0, -0.105,
        "Every registered cohort-age row is shown; diagnostics do not "
        "substitute for the primary claim.",
        transform=ax.transAxes, ha="right", va="top", fontsize=6.15,
        color=MID)
    fig.subplots_adjust(left=0.27, right=0.995, bottom=0.12, top=0.77)
    return _figure_bytes(
        fig, ledger_sha256=ledger_sha256, script_sha256=script_sha256)


def _channel_interval(row: Mapping[str, Any], channel: str
                      ) -> Mapping[str, Any] | None:
    if channel == "primary_host":
        return row[channel]["host_full_vs_locked_comparator"]
    if channel == "prior_diagnostic":
        return row[channel]["vs_locked_comparator"]
    if channel == "raw_context":
        return row[channel]["long16_minus_short3"]
    execution = row["execution"]
    return (execution["full_vs_locked_comparator"]
            if execution is not None else None)


def render_effects(plot_data: Mapping[str, Any], *, ledger_sha256: str,
                   script_sha256: str) -> tuple[bytes, bytes]:
    rows = plot_data["effect_rows"]
    channels = (
        ("primary_host", "Frozen host", "vs locked"),
        ("prior_diagnostic", "Carrier prior", "vs locked"),
        ("raw_context", "Raw context", "long - short"),
        ("execution", "Executed use", "vs locked"),
    )
    all_bounds = [
        abs(100.0 * interval[key])
        for row in rows
        for channel, _, _ in channels
        for interval in [_channel_interval(row, channel)]
        if interval is not None
        for key in ("lower", "upper")
    ]
    largest = max(all_bounds, default=5.0)
    limit = max(5.0, math.ceil(largest / 5.0) * 5.0)
    thresholds = plot_data["thresholds"]
    decision_margins = {
        "primary_host": 100.0 * thresholds["host_gain"],
        "prior_diagnostic": 0.0,
        "raw_context": 0.0,
        "execution": 100.0 * thresholds["execution_gain"],
    }

    fig, axes = plt.subplots(
        1, 4, figsize=(7.05, 4.05), sharey=True,
        gridspec_kw={"wspace": 0.10})
    y = np.arange(len(rows))
    for axis_index, (ax, (channel, title, contrast_label)) in enumerate(zip(
            axes, channels, strict=True)):
        color = CHANNEL_COLORS[channel]
        marker = CHANNEL_MARKERS[channel]
        ax.axvline(0.0, color=MID, linestyle=(0, (2.0, 2.0)),
                   linewidth=0.7, zorder=0)
        margin = decision_margins[channel]
        if margin > 0.0:
            ax.axvline(margin, color=color, linestyle=(0, (1.0, 1.5)),
                       linewidth=0.8, zorder=0)
        ax.grid(axis="x", color=GRID, linewidth=0.45, alpha=0.8, zorder=0)
        for row_index, row in enumerate(rows):
            interval = _channel_interval(row, channel)
            if interval is None:
                ax.text(0.97, row_index, "n/a", transform=ax.get_yaxis_transform(),
                        ha="right", va="center", color=NA, fontsize=5.7)
                continue
            point = 100.0 * interval["point"]
            lower = 100.0 * interval["lower"]
            upper = 100.0 * interval["upper"]
            ax.plot([lower, upper], [row_index, row_index], color=color,
                    linewidth=1.05, solid_capstyle="round", zorder=2)
            ax.scatter(point, row_index, s=15, color=color, marker=marker,
                       edgecolors="white", linewidths=0.35, zorder=3)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        tick_labels = _effect_tick_labels(axis_index, len(channels), limit)
        ax.set_xticks((-limit, 0.0, limit), tick_labels)
        ax.set_title(title, color=color, fontweight="bold", pad=22.0)
        ax.tick_params(axis="both", length=2.2, width=0.55)
        ax.spines["left"].set_visible(axis_index == 0)
        if axis_index > 0:
            ax.tick_params(axis="y", left=False, labelleft=False)
        margin_label = "gate > 0" if margin == 0.0 else (
            f"gate >= {margin:g} pp")
        ax.text(0.5, 1.012, f"{contrast_label}  |  {margin_label}",
                transform=ax.transAxes,
                ha="center", va="bottom", fontsize=5.65, color=MID)
        for boundary in (2.5, 5.5, 8.5, 11.5):
            ax.axhline(boundary, color=GRID, linewidth=0.55, zorder=0)
    axes[0].set_yticks(y, _row_labels(rows))
    for label in axes[0].get_yticklabels():
        label.set_horizontalalignment("right")
    fig.supxlabel("Effect (percentage points; 95% paired CI)",
                  x=0.64, y=0.072, fontsize=6.8)
    legend = [
        Line2D([0], [0], color=INK, marker="o", markersize=3.7,
               linewidth=1.0, label="Point and 95% CI"),
        Line2D([0], [0], color=MID, linestyle=(0, (2.0, 2.0)),
               linewidth=0.8, label="Zero effect"),
        Line2D([0], [0], color=MID, linestyle=(0, (1.0, 1.5)),
               linewidth=0.8, label="Registered margin"),
        Line2D([0], [0], color=NA, marker="_", markersize=7,
               markeredgewidth=1.2, linewidth=0, label="Not evaluated"),
    ]
    fig.legend(handles=legend, ncol=4, frameon=False, loc="lower center",
               bbox_to_anchor=(0.62, 0.015), columnspacing=1.1,
               handletextpad=0.45)
    fig.text(
        0.995, 0.985,
        "All 15 registered cohort-age rows; common symmetric scale",
        ha="right", va="top", fontsize=5.9, color=MID)
    fig.subplots_adjust(left=0.285, right=0.995, bottom=0.16, top=0.84)
    return _figure_bytes(
        fig, ledger_sha256=ledger_sha256, script_sha256=script_sha256)


def _effect_tick_labels(axis_index: int, axis_count: int,
                        limit: float) -> tuple[str, str, str]:
    """Keep adjacent panel-edge tick labels from colliding."""

    _require(0 <= axis_index < axis_count and axis_count > 0,
             "effect-axis index is invalid")
    left = f"{-limit:g}" if axis_index == 0 else ""
    right = f"{limit:g}" if axis_index == axis_count - 1 else ""
    return left, "0", right


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_symlink_components(path: Path, *, label: str) -> None:
    """Reject every existing symlink component before publication I/O."""

    absolute = _lexical_absolute(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise SageMemPlotError(
                f"{label} contains a symlink component: {current}")


def _safe_output_dir(path: Path) -> Path:
    """Confine publication output to this repository's ``paper_a`` tree."""

    repository = _lexical_absolute(ROOT)
    paper_root = repository / "paper_a"
    candidate = _lexical_absolute(path)
    try:
        candidate.relative_to(paper_root)
    except ValueError as error:
        raise SageMemPlotError(
            "--output-dir must remain inside ROOT/paper_a") from error
    _reject_symlink_components(candidate, label="publication output path")
    return candidate


def _display_path(path: Path) -> str:
    resolved = _lexical_absolute(path)
    try:
        return str(resolved.relative_to(_lexical_absolute(ROOT)))
    except ValueError:
        return str(resolved)


def _output_paths(output_dir: Path, prefix: str) -> dict[str, Path]:
    _require(PREFIX_RE.fullmatch(prefix) is not None,
             "--prefix must be one safe filename component")
    paths = {
        "claim_ladder_pdf": output_dir / f"{prefix}_claim_ladder.pdf",
        "claim_ladder_png": output_dir / f"{prefix}_claim_ladder.png",
        "effects_pdf": output_dir / f"{prefix}_effects.pdf",
        "effects_png": output_dir / f"{prefix}_effects.png",
        "plot_data": output_dir / f"{prefix}_plot_data.json",
        "manifest": output_dir / f"{prefix}_plot_manifest.json",
    }
    resolved = [path.resolve() for path in paths.values()]
    _require(len(set(resolved)) == len(resolved),
             "publication output paths must be distinct")
    return paths


def _stage(path: Path, payload: bytes) -> Path:
    _reject_symlink_components(path.parent, label="publication output path")
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(path.parent, label="publication output path")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def _publish(artifacts: Mapping[Path, bytes], *, resume: bool) -> str:
    existing = {
        path: path.exists() or path.is_symlink() for path in artifacts
    }
    if any(existing.values()):
        _require(resume,
                 "refusing to overwrite existing publication artifacts")
        for path, present in existing.items():
            if present:
                _require(_read_stable(path, "publication artifact") ==
                         artifacts[path],
                         f"resume artifact differs from authenticated "
                         f"render: {path}")
        if all(existing.values()):
            return "validated-existing"

    missing = [path for path, present in existing.items() if not present]
    staged: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for destination in missing:
            staged.append((_stage(destination, artifacts[destination]),
                           destination))
        for temporary, destination in staged:
            os.link(temporary, destination)
            published.append(destination)
        for directory in {path.parent for path in artifacts}:
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    except BaseException:
        for path in published:
            path.unlink(missing_ok=True)
        raise
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
    return "repaired-missing" if any(existing.values()) else "created"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--expected-ledger-sha256")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    _require(not args.resume or args.execute, "--resume requires --execute")
    if not args.execute:
        paths = _output_paths(args.output_dir, args.prefix)
        print(_canonical_json({
            "schema": PLOT_MANIFEST_SCHEMA,
            "study": "sage-mem-v1",
            "preview": True,
            "ledger": _display_path(args.ledger),
            "outputs": {key: _display_path(path)
                        for key, path in paths.items()},
            "required_cohorts": list(COHORTS),
            "required_ages": list(AGES),
            "required_claim_rows": 15,
            "no_files_read": True,
            "no_outcomes_read": True,
            "no_files_written": True,
        }).decode("utf-8"), end="")
        return 0

    output_dir = _safe_output_dir(args.output_dir)
    paths = _output_paths(output_dir, args.prefix)
    _require(args.expected_ledger_sha256 is not None,
             "--expected-ledger-sha256 is required with --execute")
    expected = _digest(args.expected_ledger_sha256,
                       "expected claim-ledger SHA-256")
    ledger_payload = _read_stable(args.ledger, "claim ledger")
    ledger_sha256 = _sha256_bytes(ledger_payload)
    _require(ledger_sha256 == expected,
             "claim-ledger SHA-256 differs from expected")
    ledger = validate_ledger(
        _decode_canonical(ledger_payload, "claim ledger"))
    script_path = Path(__file__).resolve()
    script_sha256 = _sha256_bytes(
        _read_stable(script_path, "plotting script"))
    plot_data = build_plot_data(
        ledger, ledger_sha256=ledger_sha256,
        script_sha256=script_sha256)
    data_payload = _canonical_json(plot_data)
    claim_pdf, claim_png = render_claim_ladder(
        plot_data, ledger_sha256=ledger_sha256,
        script_sha256=script_sha256)
    effects_pdf, effects_png = render_effects(
        plot_data, ledger_sha256=ledger_sha256,
        script_sha256=script_sha256)
    payloads_without_manifest = {
        paths["claim_ladder_pdf"]: claim_pdf,
        paths["claim_ladder_png"]: claim_png,
        paths["effects_pdf"]: effects_pdf,
        paths["effects_png"]: effects_png,
        paths["plot_data"]: data_payload,
    }
    key_by_path = {path: key for key, path in paths.items()}
    manifest = {
        "schema": PLOT_MANIFEST_SCHEMA,
        "study": "sage-mem-v1",
        "stage": "publication-plotting",
        "status": "complete",
        "source_binding": {
            "claim_ledger": {
                "path": _display_path(args.ledger),
                "size": len(ledger_payload),
                "sha256": ledger_sha256,
                "schema": LEDGER_SCHEMA,
                "expected_sha256_verified": True,
            },
            "formal_report_sha256":
                ledger["source_binding"]["report"]["sha256"],
            "protocol_fingerprint":
                ledger["source_binding"]["protocol"]["fingerprint"],
            "plotting_script": {
                "path": _display_path(script_path),
                "size": script_path.stat().st_size,
                "sha256": script_sha256,
            },
        },
        "display_contract": {
            "registered_rows_rendered": 15,
            "registered_cohorts_rendered": 5,
            "registered_ages_rendered": list(AGES),
            "claim_ladder_columns": 9,
            "effect_channels": [
                "primary_host", "prior_diagnostic", "raw_context",
                "execution",
            ],
            "all_rows_rendered_in_every_channel": True,
            "optional_execution_missingness_rendered": True,
            "shared_symmetric_effect_axis": True,
            "pdf_is_vector": True,
            "png_dpi": 300,
            "font_family": "STIXGeneral",
            "status_encoding": {
                "pass": {"color": PASS, "shape": "open circle"},
                "fail": {"color": FAIL, "shape": "cross"},
                "not_evaluated": {"color": NA, "shape": "dash"},
            },
        },
        "artifacts": {
            key_by_path[path]: {
                "path": _display_path(path),
                "size": len(payload),
                "sha256": _sha256_bytes(payload),
            }
            for path, payload in payloads_without_manifest.items()
        },
    }
    manifest_payload = _canonical_json(manifest)
    artifacts = {
        **payloads_without_manifest,
        paths["manifest"]: manifest_payload,
    }
    publication = _publish(artifacts, resume=bool(args.resume))
    print(_canonical_json({
        "schema": PLOT_MANIFEST_SCHEMA,
        "study": "sage-mem-v1",
        "status": "complete",
        "publication": publication,
        "ledger_sha256": ledger_sha256,
        "plotting_script_sha256": script_sha256,
        "registered_rows_rendered": 15,
        "output_manifest": _display_path(paths["manifest"]),
    }).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SageMemPlotError as error:
        print(f"SAGE-Mem plotter refused: {error}", file=sys.stderr)
        raise SystemExit(2) from error
