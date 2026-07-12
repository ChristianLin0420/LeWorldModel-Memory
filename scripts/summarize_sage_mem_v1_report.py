#!/usr/bin/env python3
"""Create publication data from an authenticated SAGE-Mem v1 audit report.

This adapter reads only the aggregate formal evidence-audit report and the
public preregistered specification.  It never opens formal cell artifacts,
label registries, predictions, or checkpoints.  Preview mode reads no files.

Execution validates the complete five-cohort by three-age claim grid, derives
the registered pass flags again from the reported confidence bounds, and
emits two deterministic, source-bound artifacts:

* a canonical JSON claim ledger; and
* a TeX macro/table fragment suitable for ``paper_a``.

An audit ``status`` of ``complete`` is an integrity statement, not a positive
scientific result.  The output preserves every registered cohort/age row,
including failures and unevaluated optional execution endpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sage_mem_v1_spec import (  # noqa: E402
    DEFAULT_SPEC,
    load_spec,
    output_root,
    spec_fingerprint,
)
from scripts.run_sage_mem_v1 import (  # noqa: E402
    SageMemRunError,
    require_valid_lock,
)


ADAPTER_SCHEMA = "sage_mem_v1_paper_claim_ledger_v1"
REPORT_SCHEMA = "sage_mem_v1_formal_evidence_audit_v1"
DEFAULT_REPORT = ROOT / "outputs/sage_mem_v1/formal_audit/report.json"
DEFAULT_PHASE_B_RECEIPT = ROOT / (
    "outputs/sage_mem_v1/receipts/phase_b/reproduction_receipt.json")
DEFAULT_JSON_OUTPUT = \
    ROOT / "paper_a/generated_results/sage_mem_v1_claim_ledger.json"
DEFAULT_TEX_OUTPUT = \
    ROOT / "paper_a/generated_results/sage_mem_v1_claim_ledger.tex"
DEFAULT_PUBLICATION_ROOT = ROOT / "paper_a/generated_results"
AUDITOR_SOURCE = ROOT / "scripts/audit_sage_mem_v1_formal.py"
PHASE_B_VERIFIER_SOURCE = \
    ROOT / "scripts/audit_sage_mem_v1_phase_b_reproduction.py"
PHASE_B_SCHEMA = "sage_mem_v1_phase_b_reproduction_v1"

PHASE_B_OPERATOR_PIN_KEYS = {
    "verifier_source_sha256", "protocol_lock_sha256",
    "phase_a_grid_sha256", "raw_context_summary_sha256",
    "label_registry_sha256", "execution_registry_sha256",
    "finalizer_summary_sha256", "finalized_cells_sha256",
    "formal_report_sha256",
}
PHASE_B_CONTRACT_IDENTITY = {
    "cohorts": [
        "lewm_reacher_color", "lewm_pusht_color", "dinowm_pusht_token",
        "dinowm_pusht_binding", "dinowm_pointmaze_goal"],
    "arms": [
        "none", "gru", "lstm", "ssm", "fixed_trust", "gdelta",
        "fixed_trust_aux", "ssm_aux", "sage_mem_full",
        "sage_mem_next_only", "sage_mem_no_exposure",
        "sage_mem_exposure_only"],
    "seeds": list(range(10)), "ages": [4, 8, 15],
    "classes": {
        "lewm_reacher_color": 4, "lewm_pusht_color": 4,
        "dinowm_pusht_token": 4, "dinowm_pusht_binding": 6,
        "dinowm_pointmaze_goal": 4},
    "formal_rows": {
        "lewm_reacher_color": 720, "lewm_pusht_color": 720,
        "dinowm_pusht_token": 960, "dinowm_pusht_binding": 960,
        "dinowm_pointmaze_goal": 1440},
    "consumer_rows": {
        "lewm_reacher_color": 480, "lewm_pusht_color": 480,
        "dinowm_pusht_token": 600, "dinowm_pusht_binding": 600,
        "dinowm_pointmaze_goal": 960},
    "variants": {
        "lewm_reacher_color": 1, "lewm_pusht_color": 1,
        "dinowm_pusht_token": 1, "dinowm_pusht_binding": 1,
        "dinowm_pointmaze_goal": 4},
    "physical_gpus": {
        "lewm_reacher_color": 0, "lewm_pusht_color": 0,
        "dinowm_pusht_token": 1, "dinowm_pusht_binding": 1,
        "dinowm_pointmaze_goal": 2},
}
PHASE_B_RECEIPT_KEYS = {
    "schema", "study", "stage", "status",
    "production_contract_verified", "report_reproducer_injected",
    "verifier_source_injected", "contract_identity",
    "contract_identity_sha256", "registered_contract_sha256",
    "outcome_values_emitted", "finalizer_prediction_helpers_called",
    "operator_pins", "authenticated_inventories",
    "independent_reproduction", "semantic_digests", "claim_boundary",
}
PHASE_B_INVENTORY_KEYS = {
    "verifier_source", "bound_input_files", "numerical_environment",
    "locked_producers_sha256", "phase_a_artifacts_sha256",
    "normalized_label_artifacts_sha256", "phase_a_cells",
    "raw_context_references", "finalized_cells",
    "execution_registry_status_sha256", "formal_report_sha256",
    "replayed_formal_report_sha256",
}
PHASE_B_BOUND_INPUT_KEYS = {
    "protocol_lock", "raw_context_summary", "label_registry",
    "execution_registry", "finalizer_summary", "formal_report",
}
PHASE_B_REPRODUCTION_KEYS = {
    "registered_consumer", "carrier_streams_reproduced",
    "raw_context_streams_reproduced",
    "eligible_execution_arrays_recomputed", "all_arrays_exact",
    "formal_report_byte_exact", "report_timestamp_normalization",
}
PHASE_B_SEMANTIC_KEYS = {
    "revealed_labels_sha256", "raw_phase_a_sha256",
    "execution_decks_sha256", "execution_receipts_sha256",
    "carrier_models_sha256", "carrier_predictions_sha256",
    "carrier_correctness_and_execution_sha256",
    "raw_predictions_and_correctness_sha256",
}

COHORTS = (
    "lewm_reacher_color",
    "lewm_pusht_color",
    "dinowm_pusht_token",
    "dinowm_pusht_binding",
    "dinowm_pointmaze_goal",
)
AGES = ("4", "8", "15")
ARMS = (
    "none", "gru", "lstm", "ssm", "fixed_trust", "gdelta",
    "fixed_trust_aux", "ssm_aux", "sage_mem_full",
    "sage_mem_next_only", "sage_mem_no_exposure",
    "sage_mem_exposure_only",
)
BASELINE_ARMS = {
    "gru", "lstm", "ssm", "fixed_trust", "gdelta",
    "fixed_trust_aux", "ssm_aux",
}
MECHANISM_CONTROLS = (
    "sage_mem_next_only",
    "sage_mem_no_exposure",
    "sage_mem_exposure_only",
    "fixed_trust_aux",
    "ssm_aux",
)
GATE_KEYS = (
    "host_vs_locked_comparator",
    "full_vs_reset",
    "full_vs_none",
    "all_mechanism_controls",
    "next_mse_noninferiority",
)
COHORT_LABELS = {
    "lewm_reacher_color": "LeWM Reacher color",
    "lewm_pusht_color": "LeWM PushT color",
    "dinowm_pusht_token": "DINO-WM PushT token",
    "dinowm_pusht_binding": "DINO-WM PushT binding",
    "dinowm_pointmaze_goal": "DINO-WM PointMaze goal",
}

REPORT_KEYS = {
    "schema", "study", "stage", "status",
    "phase_a_cells_verified", "finalized_cells_verified",
    "phase_a_grid_sha256", "identity_ledger_sha256",
    "comparators_verified", "resources_verified",
    "raw_context_references_verified", "bootstrap_draws_per_contrast",
    "cohorts", "execution_program", "prior_can_substitute_for_host_output",
    "per_age_claims_only", "pooled_cross_host_score_computed",
    "universal_success_claim_permitted",
}
COHORT_KEYS = {
    "locked_comparators", "comparator_receipt", "backend_admission",
    "resource_enforcement", "ages",
    "all_registered_ages_primary_host_claim_pass", "execution_supplied",
    "execution_pass_by_age",
}
AGE_KEYS = {
    "primary_endpoint", "host_full_accuracy",
    "host_full_vs_locked_comparator", "host_full_vs_reset",
    "host_full_vs_none", "reset_to_full_mse_ratio",
    "mechanism_controls", "next_feature_relative_excess", "gates",
    "primary_host_claim_pass", "prior_diagnostic",
    "raw_context_reference", "execution",
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
EXECUTION_PROGRAM_KEYS = {
    "optional", "eligible_cohorts", "minimum_eligible_cohorts",
    "program_claim_permitted", "per_age",
    "cross_age_conjunction_computed", "program_claim_pass",
}
EXECUTION_AGE_KEYS = {
    "eligible_cohorts", "cohorts_passing", "claim_permitted", "claim_pass",
}

SHA256_RE = re.compile(r"[0-9a-f]{64}")


class SageMemReportAdapterError(RuntimeError):
    """The report cannot be safely converted into publication data."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SageMemReportAdapterError(message)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False) + "\n"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_stable_bytes(path: Path, label: str) -> bytes:
    _require(path.is_file() and not path.is_symlink(),
             f"{label} is missing or unsafe: {path}")
    before = path.stat()
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise SageMemReportAdapterError(f"cannot read {label}: {path}") \
            from error
    after = path.stat()
    _require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        f"{label} changed while being read: {path}")
    return payload


def _sha256_file(path: Path, label: str) -> str:
    return _sha256_bytes(_read_stable_bytes(path, label))


def _decode_strict_json(payload: bytes, label: str) -> Any:
    def reject_duplicate_keys(
            pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    try:
        text = payload.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {token}")),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise SageMemReportAdapterError(
            f"{label} is not strict UTF-8 JSON") from error
    return value


def _read_report(path: Path) -> tuple[dict[str, Any], bytes]:
    payload = _read_stable_bytes(path, "formal audit report")
    value = _decode_strict_json(payload, f"formal audit report: {path}")
    _require(isinstance(value, dict), "formal audit report root is not a mapping")
    _require(payload == _canonical_json(value).encode("utf-8"),
             "formal audit report is not canonical auditor JSON")
    return value, payload


def _mapping(value: Any, label: str, *, keys: set[str] | None = None) \
        -> Mapping[str, Any]:
    _require(isinstance(value, dict), f"{label} must be a mapping")
    if keys is not None:
        _require(set(value) == keys,
                 f"{label} keys changed: expected {sorted(keys)}, "
                 f"observed {sorted(value)}")
    return value


def _bool(value: Any, label: str) -> bool:
    _require(isinstance(value, bool), f"{label} must be boolean")
    return value


def _integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool),
             f"{label} must be an integer")
    if minimum is not None:
        _require(value >= minimum, f"{label} is below {minimum}")
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


def _digest(value: Any, label: str) -> str:
    _require(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
             f"{label} is not a lowercase SHA-256 digest")
    return value


def _sha256_json_value(value: Any) -> str:
    """Match the Phase-B verifier's newline-free canonical JSON digest."""

    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("utf-8")
    return _sha256_bytes(payload)


def _phase_b_artifact_identity(
        value: Any, label: str, *, workspace: Path,
        expected_path: Path | None = None,
        open_artifact: bool = True) -> tuple[Path, bytes | None]:
    """Authenticate one Phase-B path/size/hash record."""

    record = _mapping(value, label, keys={"path", "size", "sha256"})
    _require(isinstance(record["path"], str) and record["path"],
             f"{label}.path is invalid")
    size = _integer(record["size"], f"{label}.size", minimum=1)
    digest = _digest(record["sha256"], f"{label}.sha256")
    raw = Path(record["path"])
    path = _lexical_absolute(raw if raw.is_absolute() else workspace / raw)
    _reject_symlink_components(path, label=label)
    if expected_path is not None:
        _require(path == _lexical_absolute(expected_path),
                 f"{label} path differs from the canonical input")
    if not open_artifact:
        return path, None
    payload = _read_stable_bytes(path, label)
    _require(len(payload) == size and _sha256_bytes(payload) == digest,
             f"{label} path/size/SHA-256 identity differs")
    return path, payload


def _phase_b_paths(
        spec: Mapping[str, Any], *, study_root: Path | None = None
        ) -> dict[str, Path]:
    root = (_lexical_absolute(study_root) if study_root is not None
            else _lexical_absolute(output_root(spec)))
    return {
        "protocol_lock": root / "protocol_lock.json",
        "raw_context_summary": root / "raw_context_phase_a/summary.json",
        "label_registry": (
            root / "formal_preparation/custody/registry.json"),
        "execution_registry": (
            root / "formal_preparation/execution_decks/registry.json"),
        "finalizer_summary": root / "formal_finalized/summary.json",
    }


def authenticate_phase_b_receipt(
        *, receipt_path: Path, expected_receipt_sha256: str,
        report_path: Path, report_bytes: bytes, report: Mapping[str, Any],
        spec: Mapping[str, Any],
        protocol_lock_binding: Mapping[str, Any],
        study_root: Path | None = None,
        nonproduction_test_fixture: bool = False) -> dict[str, Any]:
    """Authenticate the value-free independent Phase-B receipt.

    This is deliberately separate from report validation.  The publication
    adapter accepts no report unless the committed verifier receipt binds the
    exact report and every registered production input.
    """

    receipt_path = _lexical_absolute(receipt_path)
    _reject_symlink_components(
        receipt_path, label="Phase-B reproduction receipt")
    expected_receipt_sha256 = _digest(
        expected_receipt_sha256, "expected Phase-B receipt SHA-256")
    payload = _read_stable_bytes(receipt_path, "Phase-B reproduction receipt")
    _require(_sha256_bytes(payload) == expected_receipt_sha256,
             "Phase-B reproduction receipt SHA-256 differs from expected")
    receipt = _mapping(
        _decode_strict_json(payload, f"Phase-B receipt: {receipt_path}"),
        "Phase-B reproduction receipt", keys=PHASE_B_RECEIPT_KEYS)
    _require(payload == _canonical_json(receipt).encode("utf-8"),
             "Phase-B reproduction receipt is not canonical JSON")
    _require(receipt["schema"] == PHASE_B_SCHEMA
             and receipt["study"] == "sage-mem-v1"
             and receipt["stage"] == "phase-b-independent-reproduction"
             and receipt["status"] == "complete",
             "Phase-B receipt identity/status differs or is incomplete")
    _require(receipt["production_contract_verified"] is True
             and receipt["report_reproducer_injected"] is False
             and receipt["verifier_source_injected"] is False
             and receipt["outcome_values_emitted"] is False
             and receipt["finalizer_prediction_helpers_called"] is False,
             "Phase-B receipt lost its production/no-injection boundary")

    registered_digest = _sha256_json_value(PHASE_B_CONTRACT_IDENTITY)
    _require(receipt["contract_identity"] == PHASE_B_CONTRACT_IDENTITY
             and receipt["contract_identity_sha256"] == registered_digest
             and receipt["registered_contract_sha256"] == registered_digest,
             "Phase-B receipt does not bind the exact registered contract")

    pins = _mapping(
        receipt["operator_pins"], "Phase-B operator pins",
        keys=PHASE_B_OPERATOR_PIN_KEYS)
    for key, value in pins.items():
        _digest(value, f"Phase-B operator_pins.{key}")
    report_sha256 = _sha256_bytes(report_bytes)
    _require(pins["formal_report_sha256"] == report_sha256
             and pins["phase_a_grid_sha256"] ==
             report.get("phase_a_grid_sha256"),
             "Phase-B receipt does not bind the selected formal report/grid")
    lock_identity = _mapping(
        protocol_lock_binding.get("implementation_lock"),
        "protocol lock binding", keys={"path", "size", "sha256", "status"})
    _require(lock_identity["status"] == "sealed"
             and pins["protocol_lock_sha256"] == lock_identity["sha256"],
             "Phase-B receipt protocol-lock pin differs")

    inventory = _mapping(
        receipt["authenticated_inventories"],
        "Phase-B authenticated inventories", keys=PHASE_B_INVENTORY_KEYS)
    _require(_integer(inventory["phase_a_cells"],
                      "Phase-B phase_a_cells", minimum=0) == 600
             and _integer(inventory["raw_context_references"],
                          "Phase-B raw_context_references", minimum=0) == 50
             and _integer(inventory["finalized_cells"],
                          "Phase-B finalized_cells", minimum=0) == 600,
             "Phase-B authenticated inventory is incomplete")
    for key in (
            "locked_producers_sha256", "phase_a_artifacts_sha256",
            "normalized_label_artifacts_sha256",
            "execution_registry_status_sha256", "formal_report_sha256",
            "replayed_formal_report_sha256"):
        _digest(inventory[key], f"Phase-B inventories.{key}")
    _require(inventory["formal_report_sha256"] == report_sha256
             and inventory["replayed_formal_report_sha256"] == report_sha256,
             "Phase-B report reproduction is not byte exact")
    _mapping(inventory["numerical_environment"],
             "Phase-B numerical environment")

    verifier_path, _ = _phase_b_artifact_identity(
        inventory["verifier_source"], "Phase-B verifier source",
        workspace=ROOT, expected_path=PHASE_B_VERIFIER_SOURCE)
    verifier_sha256 = _sha256_file(
        verifier_path, "Phase-B verifier source")
    _require(verifier_sha256 == pins["verifier_source_sha256"],
             "Phase-B verifier source differs from its operator pin")

    paths = _phase_b_paths(spec, study_root=study_root)
    bound = _mapping(
        inventory["bound_input_files"], "Phase-B bound input files",
        keys=PHASE_B_BOUND_INPUT_KEYS)
    expected_paths = {
        **paths,
        "formal_report": _lexical_absolute(report_path),
    }
    pin_names = {
        "protocol_lock": "protocol_lock_sha256",
        "raw_context_summary": "raw_context_summary_sha256",
        "label_registry": "label_registry_sha256",
        "execution_registry": "execution_registry_sha256",
        "finalizer_summary": "finalizer_summary_sha256",
        "formal_report": "formal_report_sha256",
    }
    bound_payloads: dict[str, bytes | None] = {}
    for name in sorted(PHASE_B_BOUND_INPUT_KEYS):
        # The implementation lock was already authenticated through the
        # sealed protocol chain; re-opening it here is unnecessary and makes
        # unit fixtures less faithful to the actual trust split.
        open_artifact = name != "protocol_lock"
        _, artifact_payload = _phase_b_artifact_identity(
            bound[name], f"Phase-B bound input {name}", workspace=ROOT,
            expected_path=expected_paths[name], open_artifact=open_artifact)
        _require(bound[name]["sha256"] == pins[pin_names[name]],
                 f"Phase-B {name} identity differs from operator pin")
        bound_payloads[name] = artifact_payload
    _require(bound["protocol_lock"]["size"] == lock_identity["size"]
             and bound["protocol_lock"]["sha256"] == lock_identity["sha256"],
             "Phase-B bound protocol-lock identity differs")
    _require(bound["formal_report"]["size"] == len(report_bytes),
             "Phase-B bound formal-report size differs")

    finalizer_payload = bound_payloads["finalizer_summary"]
    assert finalizer_payload is not None
    finalizer = _mapping(
        _decode_strict_json(finalizer_payload, "Phase-B finalizer summary"),
        "Phase-B finalizer summary")
    _require(finalizer.get("schema") == "sage_mem_v1_formal_finalizer_v1"
             and finalizer.get("study") == "sage-mem-v1"
             and finalizer.get("stage") == "formal-finalizer"
             and finalizer.get("status") == "complete"
             and finalizer.get("phase_a_cells") == 600
             and finalizer.get("finalized_cells") == 600
             and finalizer.get("phase_a_grid_sha256") ==
             pins["phase_a_grid_sha256"]
             and finalizer.get("label_registry_sha256") ==
             pins["label_registry_sha256"]
             and finalizer.get("finalized_cells_sha256") ==
             pins["finalized_cells_sha256"],
             "Phase-B finalizer summary is incomplete or cross-bound wrongly")

    reproduction = _mapping(
        receipt["independent_reproduction"],
        "Phase-B independent reproduction", keys=PHASE_B_REPRODUCTION_KEYS)
    consumer = _mapping(
        reproduction["registered_consumer"],
        "Phase-B registered consumer", keys={
            "estimator", "alpha", "solver", "tol", "max_iter",
            "standardization", "carrier_models_refit",
            "raw_context_models_refit"})
    _require(consumer == {
        "estimator": "sklearn.linear_model.RidgeClassifier",
        "alpha": 1e-3, "solver": "lsqr", "tol": 1e-6,
        "max_iter": 5000,
        "standardization": "StandardScaler(mean=True,std=True)",
        "carrier_models_refit": 150, "raw_context_models_refit": 50,
    }, "Phase-B registered-consumer contract differs")
    _require(reproduction["carrier_streams_reproduced"] ==
             ["full", "reset", "prior"]
             and reproduction["raw_context_streams_reproduced"] ==
             ["short-3", "long-16"]
             and reproduction["eligible_execution_arrays_recomputed"] is True
             and reproduction["all_arrays_exact"] is True
             and reproduction["formal_report_byte_exact"] is True
             and isinstance(reproduction["report_timestamp_normalization"],
                            str)
             and reproduction["report_timestamp_normalization"],
             "Phase-B independent-reproduction assertions differ")
    semantics = _mapping(
        receipt["semantic_digests"], "Phase-B semantic digests",
        keys=PHASE_B_SEMANTIC_KEYS)
    for key, value in semantics.items():
        _digest(value, f"Phase-B semantic_digests.{key}")
    _require(receipt["claim_boundary"] == (
        "provenance-and-reproduction-only; this receipt contains no "
        "accuracy, effect, interval, gate, or universal-success claim"),
        "Phase-B value-free claim boundary changed")

    return {
        "receipt": {
            "path": _display_path(receipt_path),
            "size": len(payload),
            "sha256": expected_receipt_sha256,
            "schema": PHASE_B_SCHEMA,
            "expected_sha256_verified": True,
        },
        "verifier": {
            "path": _display_path(verifier_path),
            "size": verifier_path.stat().st_size,
            "sha256": verifier_sha256,
        },
        "registered_contract_sha256": registered_digest,
        "production_contract_verified": True,
        "report_reproducer_injected": False,
        "verifier_source_injected": False,
        "outcome_values_emitted": False,
        "operator_pins": dict(pins),
        "exact_reproduction_verified": True,
        "nonproduction_test_fixture": bool(nonproduction_test_fixture),
    }


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _lexical_absolute(path: Path) -> Path:
    """Return an absolute path without following a final symlink."""

    return Path(os.path.abspath(os.fspath(path)))


def _reject_symlink_components(path: Path, *, label: str) -> None:
    """Reject every existing symlink component in one publication path."""

    absolute = _lexical_absolute(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise SageMemReportAdapterError(
                f"{label} contains a symlink component: {current}")


def _safe_publication_output(
        path: Path, *, publication_root: Path, label: str) -> Path:
    """Confine generated ledgers to a symlink-free publication tree."""

    allowed = _lexical_absolute(publication_root)
    candidate = _lexical_absolute(path)
    _reject_symlink_components(allowed, label="publication root")
    try:
        candidate.relative_to(allowed)
    except ValueError as error:
        raise SageMemReportAdapterError(
            f"{label} must remain inside the publication root: {allowed}") \
            from error
    _reject_symlink_components(candidate, label=label)
    return candidate


def _validate_hash_fields(value: Any, label: str) -> None:
    """Validate every explicitly named SHA-256 field without opening targets."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).endswith("sha256"):
                _digest(item, f"{label}.{key}")
            _validate_hash_fields(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_hash_fields(item, f"{label}[{index}]")


def _authenticated_artifact(
        value: Any, label: str, *, relative_to: Path) -> tuple[Path, bytes]:
    """Open a report identity only after checking its exact bytes."""

    identity = _mapping(value, label, keys={"path", "size", "sha256"})
    _require(isinstance(identity["path"], str) and identity["path"],
             f"{label}.path is invalid")
    size = _integer(identity["size"], f"{label}.size", minimum=1)
    digest = _digest(identity["sha256"], f"{label}.sha256")
    raw_path = Path(identity["path"])
    path = (raw_path if raw_path.is_absolute()
            else relative_to / raw_path).resolve()
    payload = _read_stable_bytes(path, label)
    _require(len(payload) == size and _sha256_bytes(payload) == digest,
             f"{label} path/size/SHA-256 identity differs")
    return path, payload


def _bootstrap(value: Any, label: str, *, confidence: float,
               draws: int, expected_seed: int,
               lower_bound: float | None = None,
               upper_bound: float | None = None) -> dict[str, Any]:
    record = _mapping(value, label, keys=BOOTSTRAP_KEYS)
    point = _number(record["point"], f"{label}.point",
                    lower=lower_bound, upper=upper_bound)
    lower = _number(record["lower"], f"{label}.lower",
                    lower=lower_bound, upper=upper_bound)
    upper = _number(record["upper"], f"{label}.upper",
                    lower=lower_bound, upper=upper_bound)
    _require(lower <= upper, f"{label} has an inverted interval")
    observed_confidence = _number(
        record["confidence"], f"{label}.confidence", lower=0.0, upper=1.0)
    _require(math.isclose(observed_confidence, confidence,
                          rel_tol=0.0, abs_tol=1e-12),
             f"{label} confidence differs from the registered value")
    _require(_integer(record["draws"], f"{label}.draws", minimum=1) == draws,
             f"{label} draw count differs from the report contract")
    seed = _integer(record["seed"], f"{label}.seed", minimum=0)
    _require(seed == expected_seed,
             f"{label} bootstrap seed differs from the registered namespace")
    _require(record["resampling_unit"] ==
             "paired formal seed x native episode cluster",
             f"{label} resampling unit changed")
    _require(record["class_profile_stratified"] is True
             and record["pairing_preserved"] is True,
             f"{label} lost registered stratification or pairing")
    return {
        "point": point,
        "lower": lower,
        "upper": upper,
        "confidence": observed_confidence,
        "draws": draws,
        "seed": seed,
        "resampling_unit": record["resampling_unit"],
        "class_profile_stratified": True,
        "pairing_preserved": True,
    }


def _validate_comparator_receipt(
        value: Any, cohort: str, *, locked_comparators: Mapping[str, str],
        protocol_fingerprint: str, prepare_root: Path) -> dict[str, Any]:
    label = f"{cohort}.comparator_receipt"
    record = _mapping(value, label, keys={
        "formal_preparation_manifest_sha256",
        "implementation_lock_sha256",
        "custody_registry_sha256",
        "preparation_receipt",
        "locked_comparator_receipt",
    })
    for key in (
            "formal_preparation_manifest_sha256",
            "implementation_lock_sha256", "custody_registry_sha256"):
        _digest(record[key], f"{label}.{key}")
    _, preparation_payload = _authenticated_artifact(
        record["preparation_receipt"], f"{label}.preparation_receipt",
        relative_to=prepare_root)
    selection_path, selection_payload = _authenticated_artifact(
        record["locked_comparator_receipt"],
        f"{label}.locked_comparator_receipt", relative_to=prepare_root)

    preparation = _mapping(
        _decode_strict_json(preparation_payload,
                            f"{label}.preparation_receipt"),
        f"{label}.preparation_receipt")
    boundaries = _mapping(
        preparation.get("boundaries"),
        f"{label}.preparation_receipt.boundaries")
    _require(preparation.get("cohort") == cohort
             and boundaries.get("locked_comparator_receipt") ==
             record["locked_comparator_receipt"],
             f"{label} preparation receipt does not bind the selection")

    selection = _mapping(
        _decode_strict_json(selection_payload,
                            f"{label}.locked_comparator_receipt"),
        f"{label}.locked_comparator_receipt")
    observed = _mapping(
        selection.get("locked_comparators"),
        f"{label}.locked_comparator_receipt.locked_comparators",
        keys={"retention", "next_feature", "execution"})
    _require(selection.get("schema_version") == 1
             and selection.get("study") == "sage-mem-v1"
             and selection.get("stage") == "development-selection"
             and selection.get("status") == "selected"
             and selection.get("cohort") == cohort
             and selection.get("protocol_fingerprint") ==
             protocol_fingerprint
             and dict(observed) == dict(locked_comparators),
             f"{label} parsed locked comparators/protocol identity differ")
    selection_raw_path = Path(record["locked_comparator_receipt"]["path"])
    selection_expected_path = (
        selection_raw_path if selection_raw_path.is_absolute()
        else prepare_root / selection_raw_path).resolve()
    _require(selection_expected_path == selection_path,
             f"{label} selection path did not resolve identically")
    return dict(record)


def _validate_backend(value: Any, cohort: str) -> dict[str, Any]:
    label = f"{cohort}.backend_admission"
    record = _mapping(value, label)
    if cohort.startswith("lewm_"):
        expected = {
            "backend", "host_hash_before", "host_hash_after",
            "parent_overlap_zero", "formal_split_overlap_zero",
            "admission_rechecked",
        }
        _require(set(record) == expected and record["backend"] == "SIGReg-LeWM",
                 f"{label} schema/backend changed")
        before = _digest(record["host_hash_before"],
                         f"{label}.host_hash_before")
        _require(record["host_hash_after"] == before,
                 f"{label} host hash changed")
        _require(record["parent_overlap_zero"] is True
                 and record["formal_split_overlap_zero"] is True,
                 f"{label} freshness proof failed")
    else:
        expected = {
            "backend", "plan_sha256", "provenance_manifest_sha256",
            "parent_episode_overlap_count",
            "cross_split_native_episode_overlap_count", "admission_rechecked",
        }
        _require(set(record) == expected and record["backend"] == "DINO-WM",
                 f"{label} schema/backend changed")
        _digest(record["plan_sha256"], f"{label}.plan_sha256")
        _digest(record["provenance_manifest_sha256"],
                f"{label}.provenance_manifest_sha256")
        _require(_integer(record["parent_episode_overlap_count"],
                          f"{label}.parent_episode_overlap_count",
                          minimum=0) == 0
                 and _integer(
                     record["cross_split_native_episode_overlap_count"],
                     f"{label}.cross_split_native_episode_overlap_count",
                     minimum=0) == 0,
                 f"{label} freshness proof failed")
    _require(record["admission_rechecked"] is True,
             f"{label} was not rechecked")
    return dict(record)


def _validate_resources(
        value: Any, cohort: str, *, spec: Mapping[str, Any]) -> None:
    resources = _mapping(value, f"{cohort}.resource_enforcement")
    _require(set(resources) == set(ARMS),
             f"{cohort} resource ledger omits or adds arms")
    target = _integer(
        _mapping(spec["cohorts"], "cohorts")[cohort]["target_parameters"],
        f"cohorts.{cohort}.target_parameters", minimum=1)
    reporting = _mapping(spec["fairness_reporting"], "fairness_reporting")
    parameter_margin = _number(
        reporting["maximum_parameter_relative_gap"],
        "fairness_reporting.maximum_parameter_relative_gap", lower=0.0)
    flop_margin = _number(
        reporting["maximum_flop_relative_gap"],
        "fairness_reporting.maximum_flop_relative_gap", lower=0.0)
    forwards: dict[str, float] = {}
    for arm in ARMS:
        label = f"{cohort}.resource_enforcement.{arm}"
        record = _mapping(resources[arm], label)
        if arm == "none":
            _require(set(record) == {
                "trainable_parameters", "forward_flops_per_episode",
                "persistent_state_floats", "parameter_matched", "flop_matched",
            }, f"{label} schema changed")
            _require(_integer(record["trainable_parameters"],
                              f"{label}.trainable_parameters", minimum=0) == 0
                     and _integer(record["persistent_state_floats"],
                                  f"{label}.persistent_state_floats",
                                  minimum=0) == 0
                     and record["parameter_matched"] is None
                     and record["flop_matched"] is None,
                     f"{label} no-state resource contract changed")
            _number(record["forward_flops_per_episode"],
                    f"{label}.forward_flops_per_episode", lower=0.0)
            continue
        _require(set(record) == {
            "trainable_parameters", "target_parameters",
            "parameter_relative_gap", "parameter_matched",
            "forward_flops_per_episode", "persistent_state_floats",
            "peak_cuda_bytes_mean", "wall_clock_train_seconds_mean",
            "baseline_median_flops", "flop_relative_gap", "flop_matched",
        }, f"{label} schema changed")
        count = _integer(record["trainable_parameters"],
                         f"{label}.trainable_parameters", minimum=1)
        reported_target = _integer(
            record["target_parameters"], f"{label}.target_parameters",
            minimum=1)
        _require(reported_target == target,
                 f"{label} target differs from the sealed protocol")
        _integer(record["persistent_state_floats"],
                 f"{label}.persistent_state_floats", minimum=1)
        observed_parameter_gap = _number(
            record["parameter_relative_gap"],
            f"{label}.parameter_relative_gap", lower=0.0)
        expected_parameter_gap = abs(count - target) / target
        _require(math.isclose(observed_parameter_gap,
                              expected_parameter_gap,
                              rel_tol=0.0, abs_tol=1e-12),
                 f"{label} parameter gap was not derived from counts")
        _require(expected_parameter_gap <= parameter_margin
                 and record["parameter_matched"] is True,
                 f"{label} is not parameter matched")
        for key in ("peak_cuda_bytes_mean",
                    "wall_clock_train_seconds_mean"):
            _number(record[key], f"{label}.{key}", lower=0.0)
        forwards[arm] = _number(
            record["forward_flops_per_episode"],
            f"{label}.forward_flops_per_episode", lower=0.0)

    baseline_median = float(statistics.median(
        forwards[arm] for arm in ARMS if arm in BASELINE_ARMS))
    _require(baseline_median > 0.0,
             f"{cohort} baseline FLOP median is not positive")
    for arm in ARMS:
        if arm == "none":
            continue
        label = f"{cohort}.resource_enforcement.{arm}"
        record = resources[arm]
        reported_median = _number(
            record["baseline_median_flops"],
            f"{label}.baseline_median_flops", lower=0.0)
        _require(math.isclose(reported_median, baseline_median,
                              rel_tol=0.0, abs_tol=1e-12),
                 f"{label} baseline FLOP median is inconsistent")
        expected_gap = abs(forwards[arm] - baseline_median) / baseline_median
        observed_gap = _number(record["flop_relative_gap"],
                               f"{label}.flop_relative_gap", lower=0.0)
        _require(math.isclose(observed_gap, expected_gap,
                              rel_tol=0.0, abs_tol=1e-12),
                 f"{label} FLOP gap was not derived from the median")
        expected_matched = expected_gap <= flop_margin
        _require(record["flop_matched"] is expected_matched
                 and expected_matched,
                 f"{label} is not FLOP matched")


def _registered_thresholds(spec: Mapping[str, Any]) -> dict[str, float]:
    gates = _mapping(spec.get("confirmatory_gates"), "confirmatory_gates")
    return {
        "host_gain": float(gates["host_output_exposure"][
            "minimum_absolute_gain"]),
        "reset_gain": float(gates["reset_causality"][
            "minimum_absolute_drop"]),
        "reset_mse_ratio_max": float(gates["reset_causality"][
            "reset_to_full_mse_ratio_max"]),
        "mechanism_gain": float(gates["mechanism_controls"][
            "minimum_absolute_gain"]),
        "mse_relative_margin": float(gates["next_feature_noninferiority"][
            "relative_margin"]),
        "execution_gain": float(gates["execution"][
            "minimum_absolute_success_gain"]),
        "execution_oracle_gate": float(gates["execution"]["oracle_gate"]),
    }


def _authenticate_protocol_lock(
        spec: Mapping[str, Any], spec_path: Path) -> dict[str, Any]:
    """Authenticate public protocol inputs without opening formal outcomes."""

    try:
        lock = require_valid_lock(spec)
    except (OSError, ValueError, SageMemRunError) as error:
        raise SageMemReportAdapterError(
            "sealed SAGE-Mem implementation lock is invalid") from error
    lock_relative = Path(str(spec["implementation_lock"]))
    lock_path = (ROOT / lock_relative).resolve()
    _require(ROOT == lock_path or ROOT in lock_path.parents,
             "implementation lock leaves the repository")
    _require(lock_path.is_file() and not lock_path.is_symlink(),
             "implementation lock is missing or unsafe")
    lock_sha256 = _sha256_file(lock_path, "sealed implementation lock")
    fingerprint = spec_fingerprint(spec)
    _require(lock.get("schema_version") == 1
             and lock.get("study") == "sage-mem-v1"
             and lock.get("stage") == "seal"
             and lock.get("status") == "sealed"
             and lock.get("protocol_fingerprint") == fingerprint
             and lock.get("spec_sha256") == spec.get("_spec_sha256"),
             "sealed implementation lock identity differs from protocol")
    amendment = _mapping(lock.get("formal_amendment"),
                         "implementation_lock.formal_amendment")
    _require(set(amendment) == {"path", "size", "sha256", "status"}
             and amendment["status"] ==
             "locked-before-development-selection-or-formal-data",
             "sealed formal amendment identity changed")
    _digest(amendment["sha256"],
            "implementation_lock.formal_amendment.sha256")
    _integer(amendment["size"],
             "implementation_lock.formal_amendment.size", minimum=1)
    _require(isinstance(amendment["path"], str) and amendment["path"],
             "implementation lock amendment path is invalid")
    producers = _mapping(lock.get("producer_identities"),
                         "implementation_lock.producer_identities")
    auditor_identity = _mapping(
        producers.get("scripts/audit_sage_mem_v1_formal.py"),
        "implementation_lock.formal_auditor",
        keys={"size", "sha256"})
    _integer(auditor_identity["size"],
             "implementation_lock.formal_auditor.size", minimum=1)
    _digest(auditor_identity["sha256"],
            "implementation_lock.formal_auditor.sha256")
    _require(auditor_identity["sha256"] ==
             _sha256_file(AUDITOR_SOURCE, "formal evidence auditor source"),
             "current formal auditor differs from sealed source")
    spec_resolved = spec_path.resolve()
    return {
        "implementation_lock": {
            "path": _display_path(lock_path),
            "size": lock_path.stat().st_size,
            "sha256": lock_sha256,
            "status": "sealed",
        },
        "formal_amendment": dict(amendment),
        "formal_auditor": {
            "path": _display_path(AUDITOR_SOURCE),
            "size": auditor_identity["size"],
            "sha256": auditor_identity["sha256"],
            "sealed_by_implementation_lock": True,
        },
        "spec": {
            "path": _display_path(spec_resolved),
            "size": spec_resolved.stat().st_size,
            "sha256": _sha256_file(spec_resolved, "SAGE-Mem protocol"),
            "fingerprint": fingerprint,
        },
    }


def _standard_formal_roots(spec: Mapping[str, Any]) -> dict[str, Path]:
    root = output_root(spec).resolve()
    return {
        "phase_a": root,
        "finalized": root / "formal_finalized",
        "preparation": root / "formal_preparation",
        "raw_context": root / "raw_context_phase_a",
    }


def _recompute_sealed_formal_audit(
        spec: Mapping[str, Any]) -> Mapping[str, Any]:
    """Re-run the sealed auditor over the standard complete formal roots."""

    from scripts.audit_sage_mem_v1_formal import (
        FormalEvidenceAuditError,
        audit_formal_evidence,
    )

    roots = _standard_formal_roots(spec)
    try:
        return audit_formal_evidence(
            spec=spec,
            phase_a_root=roots["phase_a"],
            finalized_root=roots["finalized"],
            prepare_root=roots["preparation"],
            raw_context_root=roots["raw_context"],
        )
    except (OSError, ValueError, FormalEvidenceAuditError) as error:
        raise SageMemReportAdapterError(
            "independent sealed formal-audit recomputation failed") from error


def _validate_execution(value: Any, label: str, *, draws: int,
                        thresholds: Mapping[str, float],
                        namespace: int) -> dict[str, Any]:
    record = _mapping(value, label, keys=EXECUTION_KEYS)
    contrast_names = (
        "full_vs_locked_comparator", "full_vs_none", "full_vs_random")
    contrasts = {
        key: _bootstrap(
            record[key], f"{label}.{key}", confidence=0.95, draws=draws,
            expected_seed=namespace + 40 + index,
            lower_bound=-1.0, upper_bound=1.0)
        for index, key in enumerate(contrast_names)
    }
    _require(record["random_reference"] ==
             "sealed per-episode arm-blind random-success deck"
             and record["random_reference_is_cohort_rate"] is False,
             f"{label} random reference contract changed")
    oracle = _number(record["oracle_success"], f"{label}.oracle_success",
                     lower=0.0, upper=1.0)
    _require(oracle >= thresholds["execution_oracle_gate"],
             f"{label} bypasses the execution oracle gate")
    random_mean = _number(record["random_success_mean"],
                          f"{label}.random_success_mean", lower=0.0, upper=1.0)
    expected_pass = all(
        contrast["lower"] >= thresholds["execution_gain"]
        for contrast in contrasts.values())
    _require(_bool(record["pass"], f"{label}.pass") == expected_pass,
             f"{label} pass flag differs from registered contrasts")
    return {
        **contrasts,
        "random_reference": record["random_reference"],
        "random_reference_is_cohort_rate": False,
        "oracle_success": oracle,
        "random_success_mean": random_mean,
        "pass": expected_pass,
    }


def _validate_age(value: Any, *, cohort: str, age: str, draws: int,
                  thresholds: Mapping[str, float], namespace: int) \
        -> dict[str, Any]:
    label = f"cohorts.{cohort}.ages.{age}"
    record = _mapping(value, label, keys=AGE_KEYS)
    _require(record["primary_endpoint"] == "frozen-host full correctness",
             f"{label} primary endpoint changed")
    host_accuracy = _bootstrap(
        record["host_full_accuracy"], f"{label}.host_full_accuracy",
        confidence=0.95, draws=draws, expected_seed=namespace,
        lower_bound=0.0, upper_bound=1.0)
    host = _bootstrap(
        record["host_full_vs_locked_comparator"],
        f"{label}.host_full_vs_locked_comparator", confidence=0.95,
        draws=draws, expected_seed=namespace + 2,
        lower_bound=-1.0, upper_bound=1.0)
    reset = _bootstrap(
        record["host_full_vs_reset"], f"{label}.host_full_vs_reset",
        confidence=0.95, draws=draws, expected_seed=namespace + 3,
        lower_bound=-1.0, upper_bound=1.0)
    none = _bootstrap(
        record["host_full_vs_none"], f"{label}.host_full_vs_none",
        confidence=0.95, draws=draws, expected_seed=namespace + 4,
        lower_bound=-1.0, upper_bound=1.0)
    ratio = _number(record["reset_to_full_mse_ratio"],
                    f"{label}.reset_to_full_mse_ratio", lower=0.0)
    controls_value = _mapping(
        record["mechanism_controls"], f"{label}.mechanism_controls")
    _require(set(controls_value) == set(MECHANISM_CONTROLS),
             f"{label} mechanism-control set changed")
    controls = {
        arm: _bootstrap(controls_value[arm],
                        f"{label}.mechanism_controls.{arm}", confidence=0.95,
                        draws=draws,
                        expected_seed=namespace + 10 + index,
                        lower_bound=-1.0, upper_bound=1.0)
        for index, arm in enumerate(MECHANISM_CONTROLS)
    }
    mse = _bootstrap(
        record["next_feature_relative_excess"],
        f"{label}.next_feature_relative_excess", confidence=0.90,
        draws=draws, expected_seed=namespace + 20,
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
            mse["upper"] <= thresholds["mse_relative_margin"],
    }
    gates_value = _mapping(record["gates"], f"{label}.gates")
    _require(set(gates_value) == set(GATE_KEYS),
             f"{label} gate set changed")
    observed_gates = {
        key: _bool(gates_value[key], f"{label}.gates.{key}")
        for key in GATE_KEYS
    }
    _require(observed_gates == expected_gates,
             f"{label} gate flags differ from registered confidence bounds")
    primary_pass = all(expected_gates.values())
    _require(_bool(record["primary_host_claim_pass"],
                   f"{label}.primary_host_claim_pass") == primary_pass,
             f"{label} primary pass flag is inconsistent")

    prior_value = _mapping(record["prior_diagnostic"],
                           f"{label}.prior_diagnostic", keys={
                               "role", "accuracy", "vs_locked_comparator",
                               "resolved_positive", "enters_primary_host_claim",
                           })
    _require(prior_value["role"] ==
             "diagnostic-only; cannot establish host use"
             and prior_value["enters_primary_host_claim"] is False,
             f"{label} prior diagnostic crossed the claim boundary")
    prior_accuracy = _bootstrap(
        prior_value["accuracy"], f"{label}.prior_diagnostic.accuracy",
        confidence=0.95, draws=draws, expected_seed=namespace + 1,
        lower_bound=0.0, upper_bound=1.0)
    prior_comparator = _bootstrap(
        prior_value["vs_locked_comparator"],
        f"{label}.prior_diagnostic.vs_locked_comparator", confidence=0.95,
        draws=draws, expected_seed=namespace + 5,
        lower_bound=-1.0, upper_bound=1.0)
    prior_positive = prior_comparator["lower"] > 0.0
    _require(_bool(prior_value["resolved_positive"],
                   f"{label}.prior_diagnostic.resolved_positive") ==
             prior_positive,
             f"{label} prior diagnostic flag is inconsistent")

    raw_value = _mapping(record["raw_context_reference"],
                         f"{label}.raw_context_reference", keys={
                             "short3_accuracy", "long16_accuracy",
                             "long16_minus_short3", "resolved_long_context_gain",
                             "separate_from_parameter_matched_grid",
                         })
    short_accuracy = _number(
        raw_value["short3_accuracy"],
        f"{label}.raw_context_reference.short3_accuracy", lower=0.0, upper=1.0)
    long_accuracy = _number(
        raw_value["long16_accuracy"],
        f"{label}.raw_context_reference.long16_accuracy", lower=0.0, upper=1.0)
    raw_contrast = _bootstrap(
        raw_value["long16_minus_short3"],
        f"{label}.raw_context_reference.long16_minus_short3",
        confidence=0.95, draws=draws, expected_seed=namespace + 30,
        lower_bound=-1.0, upper_bound=1.0)
    raw_positive = raw_contrast["lower"] > 0.0
    _require(raw_value["separate_from_parameter_matched_grid"] is True
             and _bool(raw_value["resolved_long_context_gain"],
                       f"{label}.raw_context_reference.resolved_long_context_gain")
             == raw_positive,
             f"{label} raw-context diagnostic is inconsistent")

    execution = None
    if record["execution"] is not None:
        execution = _validate_execution(
            record["execution"], f"{label}.execution", draws=draws,
            thresholds=thresholds, namespace=namespace)

    return {
        "cohort": cohort,
        "cohort_label": COHORT_LABELS[cohort],
        "age": int(age),
        "primary_endpoint": record["primary_endpoint"],
        "host_full_accuracy": host_accuracy,
        "host_full_vs_locked_comparator": host,
        "host_full_vs_reset": reset,
        "host_full_vs_none": none,
        "reset_to_full_mse_ratio": ratio,
        "mechanism_controls": controls,
        "next_feature_relative_excess": mse,
        "gates": expected_gates,
        "primary_host_claim_pass": primary_pass,
        "prior_diagnostic": {
            "role": prior_value["role"],
            "accuracy": prior_accuracy,
            "vs_locked_comparator": prior_comparator,
            "resolved_positive": prior_positive,
            "enters_primary_host_claim": False,
        },
        "raw_context_reference": {
            "short3_accuracy": short_accuracy,
            "long16_accuracy": long_accuracy,
            "long16_minus_short3": raw_contrast,
            "resolved_long_context_gain": raw_positive,
            "separate_from_parameter_matched_grid": True,
        },
        "execution": execution,
    }


def _validate_execution_program(
        value: Any, rows: Sequence[Mapping[str, Any]],
        cohort_execution: Mapping[str, bool]) -> dict[str, Any]:
    record = _mapping(value, "execution_program", keys=EXECUTION_PROGRAM_KEYS)
    _require(record["optional"] is True
             and _integer(record["minimum_eligible_cohorts"],
                          "execution_program.minimum_eligible_cohorts",
                          minimum=0) == 2
             and record["cross_age_conjunction_computed"] is False
             and record["program_claim_pass"] is None,
             "execution program boundary changed")
    eligible = sum(cohort_execution.values())
    _require(_integer(record["eligible_cohorts"],
                      "execution_program.eligible_cohorts", minimum=0) == eligible,
             "execution program eligible-cohort count differs")
    permitted = eligible >= 2
    _require(_bool(record["program_claim_permitted"],
                   "execution_program.program_claim_permitted") == permitted,
             "execution program permission differs")
    by_age = _mapping(record["per_age"], "execution_program.per_age")
    _require(set(by_age) == set(AGES), "execution program age set changed")
    normalized: dict[str, Any] = {}
    for age in AGES:
        age_record = _mapping(
            by_age[age], f"execution_program.per_age.{age}",
            keys=EXECUTION_AGE_KEYS)
        passing = sum(
            1 for row in rows
            if str(row["age"]) == age and row["execution"] is not None
            and row["execution"]["pass"] is True)
        expected_pass = permitted and passing >= 2
        _require(_integer(
                     age_record["eligible_cohorts"],
                     f"execution_program.per_age.{age}.eligible_cohorts",
                     minimum=0) == eligible
                 and _integer(
                     age_record["cohorts_passing"],
                     f"execution_program.per_age.{age}.cohorts_passing",
                     minimum=0) == passing
                 and age_record["claim_permitted"] is permitted
                 and age_record["claim_pass"] is expected_pass,
                 f"execution program age-{age} summary is inconsistent")
        normalized[age] = {
            "eligible_cohorts": eligible,
            "cohorts_passing": passing,
            "claim_permitted": permitted,
            "claim_pass": expected_pass,
        }
    return {
        "optional": True,
        "eligible_cohorts": eligible,
        "minimum_eligible_cohorts": 2,
        "program_claim_permitted": permitted,
        "per_age": normalized,
        "cross_age_conjunction_computed": False,
        "program_claim_pass": None,
    }


def build_claim_ledger(
        report: Mapping[str, Any], *, spec: Mapping[str, Any],
        report_path: Path, report_bytes: bytes, spec_path: Path,
        protocol_lock_binding: Mapping[str, Any],
        phase_b_binding: Mapping[str, Any],
        independent_recomputation_verified: bool,
        expected_report_sha256: str,
        expected_protocol_fingerprint: str | None = None,
        ) -> dict[str, Any]:
    """Validate one report and return a complete canonical claim ledger."""

    _require(independent_recomputation_verified is True,
             "independent sealed-auditor recomputation was not verified")
    phase_b_binding = _mapping(
        phase_b_binding, "Phase-B publication binding", keys={
            "receipt", "verifier", "registered_contract_sha256",
            "production_contract_verified", "report_reproducer_injected",
            "verifier_source_injected", "outcome_values_emitted",
            "operator_pins", "exact_reproduction_verified",
            "nonproduction_test_fixture"})
    _require(phase_b_binding["production_contract_verified"] is True
             and phase_b_binding["report_reproducer_injected"] is False
             and phase_b_binding["verifier_source_injected"] is False
             and phase_b_binding["outcome_values_emitted"] is False
             and phase_b_binding["exact_reproduction_verified"] is True,
             "Phase-B publication binding is not production-exact")
    report = _mapping(report, "formal audit report", keys=REPORT_KEYS)
    _require(report["schema"] == REPORT_SCHEMA
             and report["study"] == "sage-mem-v1"
             and report["stage"] == "formal-evidence-audit"
             and report["status"] == "complete",
             "formal audit identity/status changed or is incomplete")
    _require(_integer(report["phase_a_cells_verified"],
                      "phase_a_cells_verified", minimum=0) == 600
             and _integer(report["finalized_cells_verified"],
                          "finalized_cells_verified", minimum=0) == 600,
             "formal report does not authenticate all 600 Phase-A/final cells")
    phase_hash = _digest(report["phase_a_grid_sha256"],
                         "phase_a_grid_sha256")
    identity_hash = _digest(report["identity_ledger_sha256"],
                            "identity_ledger_sha256")
    _require(_integer(report["comparators_verified"],
                      "comparators_verified", minimum=0) == 5
             and _integer(report["resources_verified"],
                          "resources_verified", minimum=0) == 600
             and _integer(report["raw_context_references_verified"],
                          "raw_context_references_verified", minimum=0) == 50,
             "formal report verification counts are incomplete")
    draws = _integer(report["bootstrap_draws_per_contrast"],
                     "bootstrap_draws_per_contrast", minimum=1)
    _require(draws == int(spec["statistics"]["bootstrap_draws"]),
             "report bootstrap count differs from the sealed protocol")
    _require(report["prior_can_substitute_for_host_output"] is False
             and report["per_age_claims_only"] is True
             and report["pooled_cross_host_score_computed"] is False
             and report["universal_success_claim_permitted"] is False,
             "formal report crossed a preregistered claim boundary")
    _validate_hash_fields(report, "formal audit report")

    protocol_fingerprint = spec_fingerprint(spec)
    if expected_protocol_fingerprint is not None:
        _digest(expected_protocol_fingerprint,
                "expected protocol fingerprint")
        _require(protocol_fingerprint == expected_protocol_fingerprint,
                 "current sealed protocol fingerprint differs from expected")
    report_sha256 = _sha256_bytes(report_bytes)
    _digest(expected_report_sha256, "expected report SHA-256")
    _require(report_sha256 == expected_report_sha256,
             "formal report SHA-256 differs from expected")

    thresholds = _registered_thresholds(spec)
    cohorts_value = _mapping(report["cohorts"], "cohorts")
    _require(set(cohorts_value) == set(COHORTS),
             "formal report does not contain exactly all five cohorts")
    rows: list[dict[str, Any]] = []
    cohort_summaries: dict[str, Any] = {}
    cohort_execution: dict[str, bool] = {}
    report_lock_hashes: set[str] = set()
    bootstrap_seed = _integer(
        spec["statistics"]["bootstrap_seed"],
        "statistics.bootstrap_seed", minimum=0)
    prepare_root = _standard_formal_roots(spec)["preparation"]
    for cohort_index, cohort in enumerate(COHORTS):
        value = _mapping(cohorts_value[cohort], f"cohorts.{cohort}",
                         keys=COHORT_KEYS)
        comparators = _mapping(
            value["locked_comparators"], f"{cohort}.locked_comparators",
            keys={"retention", "next_feature", "execution"})
        _require(all(item in BASELINE_ARMS for item in comparators.values()),
                 f"{cohort} has an invalid locked comparator")
        comparator_receipt = _validate_comparator_receipt(
            value["comparator_receipt"], cohort,
            locked_comparators=comparators,
            protocol_fingerprint=protocol_fingerprint,
            prepare_root=prepare_root)
        report_lock_hashes.add(
            comparator_receipt["implementation_lock_sha256"])
        backend = _validate_backend(value["backend_admission"], cohort)
        _validate_resources(value["resource_enforcement"], cohort, spec=spec)
        ages_value = _mapping(value["ages"], f"cohorts.{cohort}.ages")
        _require(set(ages_value) == set(AGES),
                 f"{cohort} does not preserve exactly ages 4, 8, and 15")
        cohort_rows = []
        for age_index, age in enumerate(AGES):
            namespace = (bootstrap_seed + cohort_index * 10_000
                         + age_index * 1_000)
            cohort_rows.append(_validate_age(
                ages_value[age], cohort=cohort, age=age,
                draws=draws, thresholds=thresholds,
                namespace=namespace))
        supplied = _bool(value["execution_supplied"],
                         f"{cohort}.execution_supplied")
        pass_by_age = _mapping(value["execution_pass_by_age"],
                               f"{cohort}.execution_pass_by_age")
        _require(set(pass_by_age) == set(AGES),
                 f"{cohort} execution age set changed")
        for row in cohort_rows:
            age = str(row["age"])
            _require((row["execution"] is not None) is supplied,
                     f"{cohort}/age-{age} execution presence is inconsistent")
            observed = pass_by_age[age]
            expected = (row["execution"]["pass"]
                        if row["execution"] is not None else None)
            _require(observed is expected,
                     f"{cohort}/age-{age} execution pass flag is inconsistent")
            row["locked_comparators"] = dict(comparators)
        all_primary = all(row["primary_host_claim_pass"] for row in cohort_rows)
        _require(value["all_registered_ages_primary_host_claim_pass"] is
                 all_primary,
                 f"{cohort} all-age summary is inconsistent")
        rows.extend(cohort_rows)
        cohort_execution[cohort] = supplied
        cohort_summaries[cohort] = {
            "cohort_label": COHORT_LABELS[cohort],
            "locked_comparators": dict(comparators),
            "comparator_receipt": comparator_receipt,
            "backend_admission": backend,
            "resource_enforcement_verified": True,
            "registered_age_rows": len(cohort_rows),
            "rows_passing_primary_host_claim": sum(
                row["primary_host_claim_pass"] for row in cohort_rows),
            "all_registered_ages_primary_host_claim_pass": all_primary,
            "execution_supplied": supplied,
            "execution_pass_by_age": {
                str(row["age"]): (
                    row["execution"]["pass"]
                    if row["execution"] is not None else None)
                for row in cohort_rows
            },
        }

    lock_identity = _mapping(
        protocol_lock_binding.get("implementation_lock"),
        "protocol lock binding", keys={"path", "size", "sha256", "status"})
    _require(report_lock_hashes == {lock_identity["sha256"]},
             "report comparator receipts do not bind the sealed "
             "implementation lock")
    _require(len(rows) == 15
             and [(row["cohort"], str(row["age"])) for row in rows]
             == [(cohort, age) for cohort in COHORTS for age in AGES],
             "claim ledger is not the complete deterministic 5 x 3 grid")
    execution_program = _validate_execution_program(
        report["execution_program"], rows, cohort_execution)
    passing = sum(row["primary_host_claim_pass"] for row in rows)
    adapter_path = Path(__file__).resolve()
    auditor_path = AUDITOR_SOURCE.resolve()
    spec_resolved = spec_path.resolve()
    sealed_spec = _mapping(protocol_lock_binding.get("spec"),
                           "sealed protocol binding")
    _require(sealed_spec.get("fingerprint") == protocol_fingerprint
             and sealed_spec.get("sha256") == spec.get("_spec_sha256"),
             "sealed protocol binding differs from the loaded specification")
    sealed_auditor = _mapping(protocol_lock_binding.get("formal_auditor"),
                              "sealed auditor binding")
    _require(sealed_auditor.get("sha256") ==
             _sha256_file(auditor_path, "formal evidence auditor source")
             and sealed_auditor.get("sealed_by_implementation_lock") is True,
             "formal auditor is not bound by the implementation lock")
    return {
        "schema": ADAPTER_SCHEMA,
        "study": "sage-mem-v1",
        "stage": "paper-claim-ledger",
        "status": "complete",
        "integrity_completion": {
            "status": "complete",
            "meaning": (
                "all registered report identities and counts authenticated; "
                "this is not a scientific pass"),
            "phase_a_cells_verified": 600,
            "finalized_cells_verified": 600,
            "comparators_verified": 5,
            "resources_verified": 600,
            "raw_context_references_verified": 50,
            "phase_a_grid_sha256": phase_hash,
            "identity_ledger_sha256": identity_hash,
        },
        "claim_policy": {
            "per_age_claims_only": True,
            "registered_cohorts": list(COHORTS),
            "registered_ages": [int(age) for age in AGES],
            "registered_claim_rows": 15,
            "positive_rows_may_not_be_selected_or_omitted": True,
            "prior_can_substitute_for_host_output": False,
            "pooled_cross_host_score_computed": False,
            "universal_success_claim_permitted": False,
            "thresholds": thresholds,
        },
        "scientific_result": {
            "status": "evaluated",
            "primary_claim_rows_total": 15,
            "primary_claim_rows_passing": passing,
            "primary_claim_rows_failing": 15 - passing,
            "any_primary_claim_row_passed": passing > 0,
            "all_primary_claim_rows_passed": passing == 15,
            "meaning": (
                "pass/fail counts summarize all registered cohort-by-age "
                "rows and do not authorize a pooled or universal claim"),
        },
        "cohort_summaries": cohort_summaries,
        "claim_rows": rows,
        "execution_program": execution_program,
        "source_binding": {
            "report": {
                "path": _display_path(report_path),
                "size": len(report_bytes),
                "sha256": report_sha256,
                "schema": REPORT_SCHEMA,
                "expected_sha256_verified": True,
                "independent_sealed_auditor_recomputation_verified": True,
                "standard_roots": {
                    key: _display_path(path)
                    for key, path in _standard_formal_roots(spec).items()
                },
            },
            "protocol": {
                "path": _display_path(spec_resolved),
                "size": spec_resolved.stat().st_size,
                "sha256": _sha256_file(spec_resolved, "SAGE-Mem protocol"),
                "fingerprint": protocol_fingerprint,
                "implementation_lock": dict(lock_identity),
                "formal_amendment": dict(
                    _mapping(protocol_lock_binding.get("formal_amendment"),
                             "formal amendment binding")),
                "report_schema_repeats_protocol_fingerprint": False,
                "binding_note": (
                    "all five report comparator receipts bind this sealed "
                    "implementation-lock SHA-256; that lock binds the protocol "
                    "fingerprint, spec hash, amendment, and formal auditor; the "
                    "report also binds phase-grid and identity-ledger hashes"),
            },
            "formal_auditor": {
                **dict(sealed_auditor),
            },
            "adapter": {
                "path": _display_path(adapter_path),
                "sha256": _sha256_file(adapter_path, "adapter source"),
            },
            "phase_b_reproduction": dict(phase_b_binding),
        },
    }


def _tex_status(value: bool | None) -> str:
    if value is True:
        return r"\SageMemGatePass"
    if value is False:
        return r"\SageMemGateFail"
    return r"\SageMemGateNA"


def _tex_escape(value: str) -> str:
    replacements = {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}",
    }
    return "".join(replacements.get(character, character)
                   for character in value)


def _tex_row_list(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render one source-bound, human-readable cohort/age list."""

    if not rows:
        return r"\textit{none}"
    return "; ".join(
        f"{_tex_escape(str(row['cohort_label']))} at age {int(row['age'])}"
        for row in rows)


def render_tex(ledger: Mapping[str, Any]) -> str:
    """Render a deterministic table that includes every registered row."""

    rows = ledger["claim_rows"]
    _require(len(rows) == 15, "TeX renderer requires all 15 claim rows")
    source = ledger["source_binding"]
    scientific = ledger["scientific_result"]
    primary_pass_rows = [
        row for row in rows if row["primary_host_claim_pass"] is True
    ]
    execution_program = ledger["execution_program"]
    execution_evaluated_rows = [
        row for row in rows if row["execution"] is not None
    ]
    execution_pass_rows = [
        row for row in execution_evaluated_rows
        if row["execution"]["pass"] is True
        and execution_program["per_age"][str(row["age"])][
            "claim_permitted"] is True
    ]
    primary_list = _tex_row_list(primary_pass_rows)
    execution_list = _tex_row_list(execution_pass_rows)
    if primary_pass_rows:
        primary_summary = (
            "The registered frozen-host conjunction passes for "
            f"{len(primary_pass_rows)} of 15 cohort--age rows: "
            f"{primary_list}.")
    else:
        primary_summary = (
            "No registered cohort--age row passes the complete frozen-host "
            "conjunction.")
    if execution_pass_rows:
        execution_summary = (
            "The registered external-execution conjunction passes for "
            f"{len(execution_pass_rows)} of "
            f"{len(execution_evaluated_rows)} evaluated cohort--age rows: "
            f"{execution_list}.")
    elif execution_evaluated_rows:
        execution_summary = (
            "No evaluated cohort--age row is both passing and permitted by "
            "the registered external-execution program.")
    else:
        execution_summary = (
            "The optional external-execution endpoint was not evaluated.")
    lines = [
        "% Generated by scripts/summarize_sage_mem_v1_report.py.",
        "% Integrity completion is not a positive scientific result.",
        f"% report-sha256: {source['report']['sha256']}",
        f"% protocol-fingerprint: {source['protocol']['fingerprint']}",
        f"% adapter-sha256: {source['adapter']['sha256']}",
        ("% phase-b-receipt-sha256: "
         f"{source['phase_b_reproduction']['receipt']['sha256']}"),
        ("% phase-b-verifier-sha256: "
         f"{source['phase_b_reproduction']['verifier']['sha256']}"),
        r"\newcommand{\SageMemGatePass}{\textsc{Pass}}",
        r"\newcommand{\SageMemGateFail}{\textsc{Fail}}",
        r"\newcommand{\SageMemGateNA}{\textemdash}",
        r"\newcommand{\SageMemIntegrityStatus}{\textsc{Complete}}",
        r"\newcommand{\SageMemIntegrityMeaning}{%",
        (r"All registered report identities and counts authenticated; "
         r"this is not a scientific pass.}"),
        (r"\newcommand{\SageMemPrimaryRowsTotal}{"
         f"{scientific['primary_claim_rows_total']}}}"),
        (r"\newcommand{\SageMemPrimaryRowsPassing}{"
         f"{scientific['primary_claim_rows_passing']}}}"),
        (r"\newcommand{\SageMemPrimaryRowsFailing}{"
         f"{scientific['primary_claim_rows_failing']}}}"),
        r"\newcommand{\SageMemPrimaryPassList}{%",
        f"{primary_list}}}",
        (r"\newcommand{\SageMemExecutionRowsEvaluated}{"
         f"{len(execution_evaluated_rows)}}}"),
        (r"\newcommand{\SageMemExecutionRowsPassing}{"
         f"{len(execution_pass_rows)}}}"),
        r"\newcommand{\SageMemExecutionPassList}{%",
        f"{execution_list}}}",
        r"\newcommand{\SageMemPrimaryResultSummary}{%",
        f"{primary_summary}}}",
        r"\newcommand{\SageMemExecutionResultSummary}{%",
        f"{execution_summary}}}",
        r"\newcommand{\SageMemClaimBoundary}{%",
        (r"Every result is specific to one registered cohort and evidence "
         r"age; carrier-prior and raw-context diagnostics do not substitute "
         r"for the frozen-host conjunction, and no pooled, universal, or "
         r"native-planner claim is made.}"),
        (r"\newcommand{\SageMemReportSHA}{\texttt{"
         f"{source['report']['sha256']}}}}}"),
        (r"\newcommand{\SageMemProtocolFingerprint}{\texttt{"
         f"{source['protocol']['fingerprint']}}}}}"),
        (r"\newcommand{\SageMemPhaseBReceiptSHA}{\texttt{"
         f"{source['phase_b_reproduction']['receipt']['sha256']}}}}}"),
        (r"\newcommand{\SageMemPhaseBVerifierSHA}{\texttt{"
         f"{source['phase_b_reproduction']['verifier']['sha256']}}}}}"),
        r"\newcommand{\SageMemClaimLedgerTable}{%",
        r"\label{sage-mem-v1:complete-claim-ledger}%",
        r"\begin{tabular}{llccccccc}",
        r"\hline",
        (r"Cohort & Age & Host & Reset & No state & Controls & Next MSE "
         r"& Primary & Execution \\"),
        r"\hline",
    ]
    for row in rows:
        execution = (row["execution"]["pass"]
                     if row["execution"] is not None else None)
        gates = row["gates"]
        lines.append(
            f"{_tex_escape(row['cohort_label'])} & {row['age']} & "
            f"{_tex_status(gates['host_vs_locked_comparator'])} & "
            f"{_tex_status(gates['full_vs_reset'])} & "
            f"{_tex_status(gates['full_vs_none'])} & "
            f"{_tex_status(gates['all_mechanism_controls'])} & "
            f"{_tex_status(gates['next_mse_noninferiority'])} & "
            f"{_tex_status(row['primary_host_claim_pass'])} & "
            f"{_tex_status(execution)} " + r"\\")
    lines.extend([
        r"\hline",
        r"\end{tabular}%",
        r"}",
        "",
    ])
    return "\n".join(lines)


def _stage_exclusive(path: Path, payload: bytes) -> Path:
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


def _publish_pair(json_path: Path, json_payload: bytes,
                  tex_path: Path, tex_payload: bytes, *, resume: bool) -> str:
    _require(_lexical_absolute(json_path) != _lexical_absolute(tex_path),
             "JSON and TeX outputs must be different paths")
    artifacts = (
        (json_path, json_payload, "JSON claim ledger"),
        (tex_path, tex_payload, "TeX claim ledger"),
    )
    existing = [path.exists() or path.is_symlink()
                for path, _, _ in artifacts]
    if any(existing):
        _require(resume, "refusing to overwrite existing publication artifacts")
        for present, (path, expected, label) in zip(
                existing, artifacts, strict=True):
            if not present:
                continue
            actual = _read_stable_bytes(path, label)
            _require(actual == expected,
                     f"resume {label} differs from authenticated report")
        if all(existing):
            return "validated-existing"

    missing = [artifact for present, artifact in zip(
        existing, artifacts, strict=True) if not present]
    staged: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for destination, payload, _ in missing:
            staged.append((
                _stage_exclusive(destination, payload), destination))
        for temporary, destination in staged:
            os.link(temporary, destination)
            published.append(destination)
        for directory in {path.parent for path, _, _ in artifacts}:
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
    return "repaired-missing" if any(existing) else "created"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--tex-output", type=Path, default=DEFAULT_TEX_OUTPUT)
    parser.add_argument("--expected-report-sha256")
    parser.add_argument("--phase-b-receipt", type=Path,
                        default=DEFAULT_PHASE_B_RECEIPT)
    parser.add_argument("--expected-phase-b-receipt-sha256")
    parser.add_argument("--expected-protocol-fingerprint")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(
        argv: Iterable[str] | None = None, *,
        recompute_formal_audit: Callable[
            [Mapping[str, Any]], Mapping[str, Any]] | None = None,
        publication_root: Path | None = None,
        phase_b_study_root: Path | None = None) -> int:
    args = parse_args(argv)
    _require(not args.resume or args.execute, "--resume requires --execute")
    if not args.execute:
        print(_canonical_json({
            "schema": ADAPTER_SCHEMA,
            "study": "sage-mem-v1",
            "preview": True,
            "report": _display_path(args.report),
            "phase_b_receipt": _display_path(args.phase_b_receipt),
            "spec": _display_path(args.spec),
            "json_output": _display_path(args.json_output),
            "tex_output": _display_path(args.tex_output),
            "required_phase_a_cells": 600,
            "required_finalized_cells": 600,
            "required_cohorts": list(COHORTS),
            "required_ages": [int(age) for age in AGES],
            "required_claim_rows": 15,
            "no_files_read": True,
            "no_outcomes_read": True,
            "no_files_written": True,
        }), end="")
        return 0

    _require(args.expected_report_sha256 is not None,
             "--expected-report-sha256 is required with --execute")
    _require(args.expected_phase_b_receipt_sha256 is not None,
             "--expected-phase-b-receipt-sha256 is required with --execute")
    production_execution = publication_root is None
    if production_execution:
        _require(recompute_formal_audit is None,
                 "production execution forbids an injected report "
                 "recomputer")
        _require(phase_b_study_root is None,
                 "production execution forbids a Phase-B fixture root")
        _require(_lexical_absolute(args.phase_b_receipt) ==
                 _lexical_absolute(DEFAULT_PHASE_B_RECEIPT),
                 "production execution requires the canonical Phase-B "
                 "receipt path")
    allowed_publication_root = (
        DEFAULT_PUBLICATION_ROOT if publication_root is None
        else publication_root)
    json_output = _safe_publication_output(
        args.json_output, publication_root=allowed_publication_root,
        label="JSON claim-ledger output")
    tex_output = _safe_publication_output(
        args.tex_output, publication_root=allowed_publication_root,
        label="TeX claim-ledger output")
    _require(json_output != tex_output,
             "JSON and TeX outputs must be different paths")
    expected_report_sha256 = _digest(
        args.expected_report_sha256, "expected report SHA-256")
    report, report_bytes = _read_report(args.report)
    _require(_sha256_bytes(report_bytes) == expected_report_sha256,
             "formal report SHA-256 differs from expected")
    _require(report.get("schema") == REPORT_SCHEMA
             and report.get("study") == "sage-mem-v1"
             and report.get("stage") == "formal-evidence-audit"
             and report.get("status") == "complete",
             "formal audit identity/status changed or is incomplete")
    try:
        spec = load_spec(args.spec, verify_parent_paths=False)
    except (OSError, ValueError) as error:
        raise SageMemReportAdapterError(
            f"cannot authenticate SAGE-Mem protocol: {args.spec}") from error
    protocol_lock_binding = _authenticate_protocol_lock(spec, args.spec)
    phase_b_binding = authenticate_phase_b_receipt(
        receipt_path=args.phase_b_receipt,
        expected_receipt_sha256=args.expected_phase_b_receipt_sha256,
        report_path=args.report, report_bytes=report_bytes, report=report,
        spec=spec, protocol_lock_binding=protocol_lock_binding,
        study_root=phase_b_study_root,
        nonproduction_test_fixture=not production_execution)
    recompute = recompute_formal_audit or _recompute_sealed_formal_audit
    independently_recomputed = recompute(spec)
    _require(isinstance(independently_recomputed, Mapping),
             "independent formal-audit recomputation returned no mapping")
    _require(report_bytes ==
             _canonical_json(independently_recomputed).encode("utf-8"),
             "formal report differs from independent sealed-auditor "
             "recomputation")
    ledger = build_claim_ledger(
        report, spec=spec, report_path=args.report,
        report_bytes=report_bytes, spec_path=args.spec,
        protocol_lock_binding=protocol_lock_binding,
        phase_b_binding=phase_b_binding,
        independent_recomputation_verified=True,
        expected_report_sha256=expected_report_sha256,
        expected_protocol_fingerprint=args.expected_protocol_fingerprint)
    tex = render_tex(ledger)
    ledger["publication_artifacts"] = {
        "tex": {
            "path": _display_path(tex_output),
            "sha256": _sha256_bytes(tex.encode("utf-8")),
        },
    }
    json_payload = _canonical_json(ledger).encode("utf-8")
    tex_payload = tex.encode("utf-8")
    publication = _publish_pair(
        json_output, json_payload, tex_output, tex_payload,
        resume=bool(args.resume))
    print(_canonical_json({
        "schema": ADAPTER_SCHEMA,
        "study": "sage-mem-v1",
        "status": "complete",
        "publication": publication,
        "integrity_status": ledger["integrity_completion"]["status"],
        "integrity_is_scientific_pass": False,
        "claim_rows": len(ledger["claim_rows"]),
        "claim_rows_passing": ledger["scientific_result"][
            "primary_claim_rows_passing"],
        "json_output": _display_path(json_output),
        "tex_output": _display_path(tex_output),
    }), end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SageMemReportAdapterError as error:
        print(f"SAGE-Mem report adapter refused: {error}", file=sys.stderr)
        raise SystemExit(2) from error
