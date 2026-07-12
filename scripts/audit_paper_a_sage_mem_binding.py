#!/usr/bin/env python3
"""Authenticate the SAGE-Mem v1 report-to-manuscript binding.

This is a post-report audit.  It independently re-runs the sealed formal
auditor over the canonical read-only experiment roots, re-runs the report
adapter, authenticates any explicitly supplied SAGE figure manifest, and
audits the reachable ``paper_a`` TeX source graph.  It never mutates campaign
artifacts.  In particular, it enforces three publication boundaries:

* all five cohorts at ages 4/8/15 remain visible (no positive-row selection);
* SAGE result artifacts enter the manuscript only after a registered primary
  pass or a passed, program-permitted execution endpoint; and
* neither figures nor prose turn a per-age audit into a universal, pooled, or
  native-planner claim.  If integrated evidence is mixed or negative, the
  complete 15-row ledger must be rendered in the appendix.  Positive prose
  is admitted only through generated result macros, and fresh AUX/FLS/PDF
  evidence must prove that the authenticated ledger and any figures reached
  the canonical ``paper_a/main.tex`` build.

Preview mode reads and writes no file.  ``--execute`` also forces a fresh
canonical ``paper_a/main.tex`` build and may create one receipt atomically.
``--resume`` accepts only an existing byte-identical receipt; it never
overwrites or repairs a different result.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "paper_a_sage_mem_binding_audit_v1"
REPORT_SCHEMA = "sage_mem_v1_formal_evidence_audit_v1"
LEDGER_SCHEMA = "sage_mem_v1_paper_claim_ledger_v1"
FIGURE_MANIFEST_SCHEMA = "sage_mem_v1_plot_manifest_v1"
PHASE_B_SCHEMA = "sage_mem_v1_phase_b_reproduction_v1"

DEFAULT_REPORT = Path("outputs/sage_mem_v1/formal_audit/report.json")
DEFAULT_LEDGER_JSON = Path(
    "paper_a/generated_results/sage_mem_v1_claim_ledger.json")
DEFAULT_LEDGER_TEX = Path(
    "paper_a/generated_results/sage_mem_v1_claim_ledger.tex")
DEFAULT_PHASE_B_RECEIPT = Path(
    "outputs/sage_mem_v1/receipts/phase_b/reproduction_receipt.json")
DEFAULT_MAIN_TEX = Path("paper_a/main.tex")
DEFAULT_RECEIPT = Path("outputs/paper_a_sage_mem_binding/receipt.json")
CANONICAL_MAIN_TEX = Path("paper_a/main.tex")
PLOTTER_SOURCE = Path("scripts/plot_sage_mem_v1_claims.py")
PHASE_B_VERIFIER_SOURCE = Path(
    "scripts/audit_sage_mem_v1_phase_b_reproduction.py")
TRUSTED_LOCAL_TEX_SUPPORT = {
    Path("paper_a/iclr2026_conference.sty"):
        "a4852f68e080d6c5245057ca2039100b409e31727898aa93c03d78ddb84374a3",
    Path("paper_a/natbib.sty"):
        "88bc70c0e48461934cab5b2accef06b74a8b3ac45ad03ccd3f2a6b7e0d6d530d",
}
TEX_SEARCH_ENV_KEYS = {
    "TEXINPUTS", "BIBINPUTS", "BSTINPUTS", "TEXMFCNF", "TEXMFHOME",
    "TEXMFLOCAL", "TEXMFCONFIG", "TEXMFVAR", "TEXMFSYSCONFIG",
    "TEXMFSYSVAR", "TEXFORMATS", "TEXPOOL", "TEXFONTMAPS",
    "VARTEXFONTS", "LUAINPUTS", "MFINPUTS", "MPINPUTS",
    "TEXMFOUTPUT", "TEXMFDBS", "TEXMFCASEFOLDSEARCH", "LATEXMKRC",
    "openout_any", "openin_any", "shell_escape",
}
LEDGER_SENTINEL = "sage-mem-v1:complete-claim-ledger"
PUBLICATION_RECOMPUTATION_KEYS = {
    "report_payload", "ledger_payload", "ledger_tex_payload",
}

RESULT_MACROS = {
    "SageMemPrimaryRowsTotal",
    "SageMemPrimaryRowsPassing",
    "SageMemPrimaryRowsFailing",
    "SageMemPrimaryPassList",
    "SageMemExecutionRowsEvaluated",
    "SageMemExecutionRowsPassing",
    "SageMemExecutionPassList",
    "SageMemPrimaryResultSummary",
    "SageMemExecutionResultSummary",
}
BOUNDARY_MACROS = {
    "SageMemClaimBoundary", "SageMemIntegrityStatus",
    "SageMemIntegrityMeaning", "SageMemReportSHA",
    "SageMemProtocolFingerprint", "SageMemPhaseBReceiptSHA",
    "SageMemPhaseBVerifierSHA",
}
STRUCTURAL_MACROS = {"SageMemClaimLedgerTable"}
ALLOWED_MANUSCRIPT_MACROS = (
    RESULT_MACROS | BOUNDARY_MACROS | STRUCTURAL_MACROS)

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
GATE_KEYS = {
    "host_vs_locked_comparator",
    "full_vs_reset",
    "full_vs_none",
    "all_mechanism_controls",
    "next_mse_noninferiority",
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
LEDGER_KEYS = {
    "schema", "study", "stage", "status", "integrity_completion",
    "claim_policy", "scientific_result", "cohort_summaries", "claim_rows",
    "execution_program", "source_binding", "publication_artifacts",
}
ROW_REPORT_FIELDS = (
    "primary_endpoint", "host_full_accuracy",
    "host_full_vs_locked_comparator", "host_full_vs_reset",
    "host_full_vs_none", "reset_to_full_mse_ratio", "mechanism_controls",
    "next_feature_relative_excess", "gates", "primary_host_claim_pass",
    "prior_diagnostic", "raw_context_reference", "execution",
)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
PHASE_B_OPERATOR_PIN_KEYS = {
    "verifier_source_sha256", "protocol_lock_sha256",
    "phase_a_grid_sha256", "raw_context_summary_sha256",
    "label_registry_sha256", "execution_registry_sha256",
    "finalizer_summary_sha256", "finalized_cells_sha256",
    "formal_report_sha256",
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
PHASE_B_CONTRACT_IDENTITY = {
    "cohorts": list(COHORTS),
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


class SageMemBindingAuditError(RuntimeError):
    """A report, publication artifact, or claim boundary is invalid."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SageMemBindingAuditError(message)


def _load_claim_plotter() -> Any:
    """Import the heavy renderer only during an executed figure audit."""

    return importlib.import_module("scripts.plot_sage_mem_v1_claims")


def _load_report_adapter() -> Any:
    """Import the sealed-report adapter only during an executed audit."""

    return importlib.import_module("scripts.summarize_sage_mem_v1_report")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False) + "\n"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _digest(value: Any, label: str) -> str:
    require(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
            f"{label} is not a lowercase SHA-256 digest")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    require(isinstance(value, int) and not isinstance(value, bool)
            and value >= minimum, f"{label} is not an integer >= {minimum}")
    return value


def _lexical_absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_symlink_components(path: Path, *, label: str) -> None:
    absolute = _lexical_absolute(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise SageMemBindingAuditError(
                f"{label} contains a symlink component: {current}")


def repository_path(root: Path, value: str | Path) -> Path:
    base = _lexical_absolute(root)
    raw = Path(value)
    candidate = (_lexical_absolute(raw) if raw.is_absolute()
                 else _lexical_absolute(base / raw))
    try:
        candidate.relative_to(base)
    except ValueError as error:
        raise SageMemBindingAuditError(f"path leaves repository: {value}") \
            from error
    _reject_symlink_components(candidate, label="repository path")
    return candidate


def display_path(root: Path, path: Path) -> str:
    resolved = _lexical_absolute(path)
    try:
        return str(resolved.relative_to(_lexical_absolute(root)))
    except ValueError:
        # Synthetic-root tests still bind the real, executing auditor.  The
        # production repository path remains relative; an external path is
        # represented explicitly rather than silently rebased.
        return str(resolved)


def read_stable_bytes(path: Path, label: str) -> bytes:
    _reject_symlink_components(path, label=label)
    require(path.is_file() and not path.is_symlink(),
            f"missing or unsafe {label}: {path}")
    before = path.stat()
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise SageMemBindingAuditError(f"cannot read {label}: {path}") \
            from error
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            f"{label} changed while being read: {path}")
    return payload


def sha256_file(path: Path, label: str = "file") -> str:
    return sha256_bytes(read_stable_bytes(path, label))


def decode_strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def object_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {token}")),
            object_pairs_hook=object_pairs,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise SageMemBindingAuditError(
            f"{label} is not strict UTF-8 JSON") from error
    require(isinstance(value, dict), f"{label} root is not a mapping")
    return value


def read_canonical_json(path: Path, label: str) \
        -> tuple[dict[str, Any], bytes]:
    payload = read_stable_bytes(path, label)
    value = decode_strict_json(payload, label)
    require(payload == canonical_json(value).encode("utf-8"),
            f"{label} is not canonical JSON")
    return value, payload


def recompute_authenticated_publication(
        root: Path, report_path: Path, ledger_path: Path,
        ledger_tex_path: Path,
        phase_b_binding: Mapping[str, Any]) -> dict[str, bytes]:
    """Re-run the sealed audit and adapter over canonical production roots."""

    root = _lexical_absolute(root)
    require(root == _lexical_absolute(ROOT),
            "production publication recomputation requires repository ROOT")
    adapter = _load_report_adapter()
    try:
        spec_path = Path(adapter.DEFAULT_SPEC)
        spec = adapter.load_spec(spec_path, verify_parent_paths=False)
        protocol_binding = adapter._authenticate_protocol_lock(spec, spec_path)
        report = adapter._recompute_sealed_formal_audit(spec)
        report_payload = adapter._canonical_json(report).encode("utf-8")
        report_sha256 = sha256_bytes(report_payload)
        ledger = adapter.build_claim_ledger(
            report, spec=spec, report_path=report_path,
            report_bytes=report_payload, spec_path=spec_path,
            protocol_lock_binding=protocol_binding,
            phase_b_binding=phase_b_binding,
            independent_recomputation_verified=True,
            expected_report_sha256=report_sha256)
        ledger_tex_payload = adapter.render_tex(ledger).encode("utf-8")
        ledger["publication_artifacts"] = {
            "tex": {
                "path": adapter._display_path(ledger_tex_path),
                "sha256": sha256_bytes(ledger_tex_payload),
            },
        }
        ledger_payload = adapter._canonical_json(ledger).encode("utf-8")
    except Exception as error:
        raise SageMemBindingAuditError(
            "sealed formal-audit/report-adapter recomputation failed") \
            from error
    return {
        "report_payload": report_payload,
        "ledger_payload": ledger_payload,
        "ledger_tex_payload": ledger_tex_payload,
    }


def authenticate_publication_recomputation(
        value: Any, *, report_payload: bytes, ledger_payload: bytes,
        ledger_tex_payload: bytes) -> None:
    """Require exact bytes from an independently executed publication chain."""

    recomputed = _mapping(
        value, "publication recomputation",
        keys=PUBLICATION_RECOMPUTATION_KEYS)
    expected = {
        "report_payload": report_payload,
        "ledger_payload": ledger_payload,
        "ledger_tex_payload": ledger_tex_payload,
    }
    for key, observed in recomputed.items():
        require(isinstance(observed, bytes),
                f"publication recomputation {key} is not bytes")
        require(observed == expected[key],
                f"{key.replace('_', ' ')} differs from independent sealed "
                "publication recomputation")


def _mapping(value: Any, label: str,
             *, keys: set[str] | None = None) -> Mapping[str, Any]:
    require(isinstance(value, dict), f"{label} must be a mapping")
    if keys is not None:
        require(set(value) == keys,
                f"{label} keys differ: observed={sorted(value)}")
    return value


def _bool(value: Any, label: str) -> bool:
    require(isinstance(value, bool), f"{label} must be boolean")
    return value


def _resolve_identity_path(root: Path, record: Mapping[str, Any],
                           label: str) -> Path:
    require(isinstance(record.get("path"), str) and record["path"],
            f"{label}.path is missing")
    return repository_path(root, record["path"])


def authenticate_identity(
        root: Path, record: Any, label: str, *, expected_path: Path | None = None,
        require_size: bool = False) -> dict[str, Any]:
    identity = _mapping(record, label)
    require("path" in identity and "sha256" in identity,
            f"{label} lacks path/SHA-256")
    path = _resolve_identity_path(root, identity, label)
    if expected_path is not None:
        require(path == _lexical_absolute(expected_path),
                f"{label} path differs")
    expected = _digest(identity["sha256"], f"{label}.sha256")
    payload = read_stable_bytes(path, label)
    if require_size or "size" in identity:
        require("size" in identity, f"{label}.size is missing")
        require(_integer(identity["size"], f"{label}.size") == len(payload),
                f"{label} size differs")
    if "bytes" in identity:
        require(_integer(identity["bytes"], f"{label}.bytes") == len(payload),
                f"{label} byte count differs")
    require(sha256_bytes(payload) == expected, f"{label} SHA-256 differs")
    return {
        "path": display_path(root, path),
        "size": len(payload),
        "sha256": expected,
    }


def _sha256_json_value(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("utf-8")
    return sha256_bytes(payload)


def authenticate_phase_b_receipt(
        root: Path, receipt_path: Path, *, expected_sha256: str,
        report_path: Path, report_payload: bytes,
        report_value: Mapping[str, Any],
        nonproduction_test_fixture: bool) -> tuple[dict[str, Any], bytes]:
    """Authenticate the committed, value-free Phase-B receipt and pins."""

    expected = _digest(expected_sha256, "expected Phase-B receipt SHA-256")
    receipt, payload = read_canonical_json(
        receipt_path, "Phase-B reproduction receipt")
    require(sha256_bytes(payload) == expected,
            "Phase-B reproduction receipt SHA-256 differs from expected")
    _mapping(receipt, "Phase-B reproduction receipt",
             keys=PHASE_B_RECEIPT_KEYS)
    require(receipt["schema"] == PHASE_B_SCHEMA
            and receipt["study"] == "sage-mem-v1"
            and receipt["stage"] == "phase-b-independent-reproduction"
            and receipt["status"] == "complete",
            "Phase-B receipt identity/status differs or is incomplete")
    require(receipt["production_contract_verified"] is True
            and receipt["report_reproducer_injected"] is False
            and receipt["verifier_source_injected"] is False
            and receipt["outcome_values_emitted"] is False
            and receipt["finalizer_prediction_helpers_called"] is False,
            "Phase-B receipt lost its production/no-injection boundary")
    contract_digest = _sha256_json_value(PHASE_B_CONTRACT_IDENTITY)
    require(receipt["contract_identity"] == PHASE_B_CONTRACT_IDENTITY
            and receipt["contract_identity_sha256"] == contract_digest
            and receipt["registered_contract_sha256"] == contract_digest,
            "Phase-B receipt registered contract differs")
    pins = _mapping(receipt["operator_pins"], "Phase-B operator pins",
                    keys=PHASE_B_OPERATOR_PIN_KEYS)
    normalized_pins = {
        key: _digest(value, f"Phase-B operator pin {key}")
        for key, value in pins.items()
    }
    report_hash = sha256_bytes(report_payload)
    require(normalized_pins["formal_report_sha256"] == report_hash,
            "Phase-B receipt does not bind the selected formal report")
    require(normalized_pins["phase_a_grid_sha256"] ==
            report_value.get("phase_a_grid_sha256"),
            "Phase-B receipt does not bind the formal report Phase-A grid")

    inventories = _mapping(
        receipt["authenticated_inventories"],
        "Phase-B authenticated inventories", keys={
            "verifier_source", "bound_input_files",
            "numerical_environment", "locked_producers_sha256",
            "phase_a_artifacts_sha256",
            "normalized_label_artifacts_sha256", "phase_a_cells",
            "raw_context_references", "finalized_cells",
            "execution_registry_status_sha256", "formal_report_sha256",
            "replayed_formal_report_sha256"})
    require(inventories["phase_a_cells"] == 600
            and inventories["raw_context_references"] == 50
            and inventories["finalized_cells"] == 600,
            "Phase-B authenticated inventory is incomplete")
    for key in (
            "locked_producers_sha256", "phase_a_artifacts_sha256",
            "normalized_label_artifacts_sha256",
            "execution_registry_status_sha256", "formal_report_sha256",
            "replayed_formal_report_sha256"):
        _digest(inventories[key], f"Phase-B inventory {key}")
    require(inventories["formal_report_sha256"] == report_hash
            and inventories["replayed_formal_report_sha256"] == report_hash,
            "Phase-B report replay identity differs")
    _mapping(inventories["numerical_environment"],
             "Phase-B numerical environment")

    verifier_path = repository_path(root, PHASE_B_VERIFIER_SOURCE)
    verifier = authenticate_identity(
        root, inventories["verifier_source"], "Phase-B verifier source",
        expected_path=verifier_path, require_size=True)
    require(verifier["sha256"] ==
            normalized_pins["verifier_source_sha256"],
            "Phase-B verifier source differs from its operator pin")

    expected_paths = {
        "protocol_lock": Path("outputs/sage_mem_v1/protocol_lock.json"),
        "raw_context_summary": Path(
            "outputs/sage_mem_v1/raw_context_phase_a/summary.json"),
        "label_registry": Path(
            "outputs/sage_mem_v1/formal_preparation/custody/registry.json"),
        "execution_registry": Path(
            "outputs/sage_mem_v1/formal_preparation/"
            "execution_decks/registry.json"),
        "finalizer_summary": Path(
            "outputs/sage_mem_v1/formal_finalized/summary.json"),
        "formal_report": report_path,
    }
    pin_names = {
        "protocol_lock": "protocol_lock_sha256",
        "raw_context_summary": "raw_context_summary_sha256",
        "label_registry": "label_registry_sha256",
        "execution_registry": "execution_registry_sha256",
        "finalizer_summary": "finalizer_summary_sha256",
        "formal_report": "formal_report_sha256",
    }
    bound = _mapping(
        inventories["bound_input_files"], "Phase-B bound inputs",
        keys=set(expected_paths))
    normalized_inputs: dict[str, Any] = {}
    for name, relative in expected_paths.items():
        expected_path = (relative if Path(relative).is_absolute()
                         else repository_path(root, relative))
        identity = authenticate_identity(
            root, bound[name], f"Phase-B bound input {name}",
            expected_path=expected_path, require_size=True)
        require(identity["sha256"] == normalized_pins[pin_names[name]],
                f"Phase-B bound input {name} differs from operator pin")
        normalized_inputs[name] = identity
    require(normalized_inputs["formal_report"]["sha256"] == report_hash,
            "Phase-B formal-report input identity differs")

    finalizer, _ = read_canonical_json(
        repository_path(root, expected_paths["finalizer_summary"]),
        "Phase-B finalizer summary")
    require(finalizer.get("schema") == "sage_mem_v1_formal_finalizer_v1"
            and finalizer.get("study") == "sage-mem-v1"
            and finalizer.get("stage") == "formal-finalizer"
            and finalizer.get("status") == "complete"
            and finalizer.get("phase_a_cells") == 600
            and finalizer.get("finalized_cells") == 600
            and finalizer.get("phase_a_grid_sha256") ==
            normalized_pins["phase_a_grid_sha256"]
            and finalizer.get("label_registry_sha256") ==
            normalized_pins["label_registry_sha256"]
            and finalizer.get("finalized_cells_sha256") ==
            normalized_pins["finalized_cells_sha256"],
            "Phase-B finalizer summary cross-binding differs")

    reproduction = _mapping(
        receipt["independent_reproduction"],
        "Phase-B independent reproduction", keys={
            "registered_consumer", "carrier_streams_reproduced",
            "raw_context_streams_reproduced",
            "eligible_execution_arrays_recomputed", "all_arrays_exact",
            "formal_report_byte_exact", "report_timestamp_normalization"})
    consumer = _mapping(
        reproduction["registered_consumer"],
        "Phase-B registered consumer", keys={
            "estimator", "alpha", "solver", "tol", "max_iter",
            "standardization", "carrier_models_refit",
            "raw_context_models_refit"})
    require(consumer == {
        "estimator": "sklearn.linear_model.RidgeClassifier",
        "alpha": 1e-3, "solver": "lsqr", "tol": 1e-6,
        "max_iter": 5000,
        "standardization": "StandardScaler(mean=True,std=True)",
        "carrier_models_refit": 150, "raw_context_models_refit": 50,
    } and reproduction["carrier_streams_reproduced"] ==
        ["full", "reset", "prior"]
        and reproduction["raw_context_streams_reproduced"] ==
        ["short-3", "long-16"]
        and reproduction["eligible_execution_arrays_recomputed"] is True
        and reproduction["all_arrays_exact"] is True
        and reproduction["formal_report_byte_exact"] is True
        and isinstance(reproduction["report_timestamp_normalization"], str)
        and bool(reproduction["report_timestamp_normalization"]),
        "Phase-B independent reproduction contract differs")
    semantics = _mapping(
        receipt["semantic_digests"], "Phase-B semantic digests", keys={
            "revealed_labels_sha256", "raw_phase_a_sha256",
            "execution_decks_sha256", "execution_receipts_sha256",
            "carrier_models_sha256", "carrier_predictions_sha256",
            "carrier_correctness_and_execution_sha256",
            "raw_predictions_and_correctness_sha256"})
    for key, value in semantics.items():
        _digest(value, f"Phase-B semantic digest {key}")
    require(receipt["claim_boundary"] == (
        "provenance-and-reproduction-only; this receipt contains no "
        "accuracy, effect, interval, gate, or universal-success claim"),
        "Phase-B claim boundary differs")

    binding = {
        "receipt": {
            "path": display_path(root, receipt_path),
            "size": len(payload), "sha256": expected,
            "schema": PHASE_B_SCHEMA,
            "expected_sha256_verified": True,
        },
        "verifier": verifier,
        "registered_contract_sha256": contract_digest,
        "production_contract_verified": True,
        "report_reproducer_injected": False,
        "verifier_source_injected": False,
        "outcome_values_emitted": False,
        "operator_pins": normalized_pins,
        "exact_reproduction_verified": True,
        "nonproduction_test_fixture": bool(nonproduction_test_fixture),
    }
    return binding, payload


def replay_phase_b_receipt(
        root: Path, receipt: Mapping[str, Any], *,
        recomputer: Callable[[Path, Mapping[str, Any]], bytes] | None = None
        ) -> bytes:
    """Re-run the committed verifier in a symlink-free temporary output."""

    if recomputer is not None:
        require(_lexical_absolute(root) != _lexical_absolute(ROOT),
                "production ROOT cannot inject a Phase-B recomputer")
        replayed = recomputer(root, receipt)
        require(isinstance(replayed, bytes),
                "injected Phase-B recomputer did not return bytes")
        return replayed

    verifier_path = repository_path(root, PHASE_B_VERIFIER_SOURCE)
    pins = receipt["operator_pins"]
    require(sha256_file(verifier_path, "committed Phase-B verifier") ==
            pins["verifier_source_sha256"],
            "committed Phase-B verifier source changed before replay")
    interpreter = _lexical_absolute(sys.executable)
    require(interpreter.is_file(), "Python replay interpreter is missing")
    with tempfile.TemporaryDirectory(
            prefix=".phase-b-publication-replay-", dir=root) as directory:
        temporary = _lexical_absolute(directory)
        _reject_symlink_components(temporary, label="Phase-B replay output")
        output = temporary / "reproduction_receipt.json"
        command = [
            str(interpreter), "-I", str(verifier_path),
            "--workspace", str(root),
            "--protocol-lock", "outputs/sage_mem_v1/protocol_lock.json",
            "--phase-a-root", "outputs/sage_mem_v1",
            "--raw-context-root",
            "outputs/sage_mem_v1/raw_context_phase_a",
            "--label-registry",
            "outputs/sage_mem_v1/formal_preparation/custody/registry.json",
            "--execution-registry",
            ("outputs/sage_mem_v1/formal_preparation/"
             "execution_decks/registry.json"),
            "--finalized-root", "outputs/sage_mem_v1/formal_finalized",
            "--prepare-root", "outputs/sage_mem_v1/formal_preparation",
            "--formal-report", "outputs/sage_mem_v1/formal_audit/report.json",
            "--output", str(output),
            "--expected-verifier-source-sha256",
            pins["verifier_source_sha256"],
            "--expected-protocol-lock-sha256",
            pins["protocol_lock_sha256"],
            "--expected-phase-a-grid-sha256",
            pins["phase_a_grid_sha256"],
            "--expected-raw-context-summary-sha256",
            pins["raw_context_summary_sha256"],
            "--expected-label-registry-sha256",
            pins["label_registry_sha256"],
            "--expected-execution-registry-sha256",
            pins["execution_registry_sha256"],
            "--expected-finalizer-summary-sha256",
            pins["finalizer_summary_sha256"],
            "--expected-finalized-cells-sha256",
            pins["finalized_cells_sha256"],
            "--expected-formal-report-sha256",
            pins["formal_report_sha256"],
            "--execute",
        ]
        environment = {
            key: value for key, value in os.environ.items()
            if key not in {
                "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP",
                "PYTHONINSPECT"}
        }
        environment["PYTHONNOUSERSITE"] = "1"
        try:
            completed = subprocess.run(
                command, cwd=root, env=environment, text=True,
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, check=False)
        except OSError as error:
            raise SageMemBindingAuditError(
                "cannot launch isolated committed Phase-B verifier") \
                from error
        require(completed.returncode == 0,
                "isolated committed Phase-B verifier replay failed: "
                + completed.stdout[-2000:])
        require(sha256_file(verifier_path, "committed Phase-B verifier") ==
                pins["verifier_source_sha256"],
                "committed Phase-B verifier source changed during replay")
        return read_stable_bytes(output, "replayed Phase-B receipt")


def authenticate_report(report: Mapping[str, Any]) \
        -> dict[tuple[str, int], Mapping[str, Any]]:
    _mapping(report, "formal report", keys=REPORT_KEYS)
    require(report["schema"] == REPORT_SCHEMA
            and report["study"] == "sage-mem-v1"
            and report["stage"] == "formal-evidence-audit"
            and report["status"] == "complete",
            "formal report identity/status differs or is incomplete")
    require(report["phase_a_cells_verified"] == 600
            and report["finalized_cells_verified"] == 600
            and report["comparators_verified"] == 5
            and report["resources_verified"] == 600
            and report["raw_context_references_verified"] == 50,
            "formal report verification counts are incomplete")
    _digest(report["phase_a_grid_sha256"], "phase_a_grid_sha256")
    _digest(report["identity_ledger_sha256"], "identity_ledger_sha256")
    _integer(report["bootstrap_draws_per_contrast"],
             "bootstrap_draws_per_contrast", minimum=1)
    require(report["prior_can_substitute_for_host_output"] is False
            and report["per_age_claims_only"] is True
            and report["pooled_cross_host_score_computed"] is False
            and report["universal_success_claim_permitted"] is False,
            "formal report crosses a preregistered claim boundary")

    cohorts = _mapping(report["cohorts"], "formal report cohorts")
    require(set(cohorts) == set(COHORTS),
            "formal report must contain exactly all five cohorts")
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    for cohort in COHORTS:
        cohort_record = _mapping(cohorts[cohort], f"report.{cohort}")
        ages = _mapping(cohort_record.get("ages"), f"report.{cohort}.ages")
        require(set(ages) == {str(age) for age in AGES},
                f"report {cohort} must contain exactly ages 4, 8, and 15")
        pass_by_age = _mapping(
            cohort_record.get("execution_pass_by_age"),
            f"report.{cohort}.execution_pass_by_age")
        require(set(pass_by_age) == set(ages),
                f"report {cohort} execution age coverage differs")
        supplied = _bool(cohort_record.get("execution_supplied"),
                         f"report.{cohort}.execution_supplied")
        primary_values: list[bool] = []
        for age in AGES:
            record = _mapping(ages[str(age)],
                              f"report.{cohort}.ages.{age}")
            gates = _mapping(record.get("gates"),
                             f"report.{cohort}.ages.{age}.gates",
                             keys=GATE_KEYS)
            observed_gates = {
                key: _bool(value, f"report.{cohort}.{age}.gates.{key}")
                for key, value in gates.items()
            }
            primary = _bool(record.get("primary_host_claim_pass"),
                            f"report.{cohort}.{age}.primary pass")
            require(primary == all(observed_gates.values()),
                    f"report {cohort}/age-{age} primary pass differs from gates")
            execution = record.get("execution")
            require((execution is not None) == supplied,
                    f"report {cohort}/age-{age} execution presence differs")
            execution_pass = None
            if execution is not None:
                execution_pass = _bool(
                    _mapping(execution, f"report.{cohort}.{age}.execution")
                    .get("pass"), f"report.{cohort}.{age}.execution.pass")
            require(pass_by_age[str(age)] is execution_pass,
                    f"report {cohort}/age-{age} execution flag differs")
            primary_values.append(primary)
            result[(cohort, age)] = record
        require(cohort_record.get(
                    "all_registered_ages_primary_host_claim_pass")
                is all(primary_values),
                f"report {cohort} all-age summary differs")

    _authenticate_execution_program(report["execution_program"], result)
    return result


def _authenticate_execution_program(
        value: Any,
        rows: Mapping[tuple[str, int], Mapping[str, Any]]) -> None:
    program = _mapping(value, "execution program")
    require(program.get("optional") is True
            and program.get("minimum_eligible_cohorts") == 2
            and program.get("cross_age_conjunction_computed") is False
            and program.get("program_claim_pass") is None,
            "execution program boundary differs")
    eligible_cohorts = sum(
        rows[(cohort, AGES[0])].get("execution") is not None
        for cohort in COHORTS)
    require(program.get("eligible_cohorts") == eligible_cohorts
            and program.get("program_claim_permitted")
            is (eligible_cohorts >= 2),
            "execution program eligibility differs")
    per_age = _mapping(program.get("per_age"), "execution program per-age")
    require(set(per_age) == {str(age) for age in AGES},
            "execution program must preserve all registered ages")
    for age in AGES:
        record = _mapping(per_age[str(age)], f"execution program age {age}")
        passing = sum(
            row.get("execution") is not None
            and row["execution"].get("pass") is True
            for (cohort_name, row_age), row in rows.items()
            if row_age == age)
        permitted = eligible_cohorts >= 2
        require(record.get("eligible_cohorts") == eligible_cohorts
                and record.get("cohorts_passing") == passing
                and record.get("claim_permitted") is permitted
                and record.get("claim_pass") is (permitted and passing >= 2),
                f"execution program age-{age} summary differs")


def authenticate_ledger(
        ledger: Mapping[str, Any], report: Mapping[str, Any],
        report_rows: Mapping[tuple[str, int], Mapping[str, Any]]) \
        -> dict[str, Any]:
    _mapping(ledger, "claim ledger", keys=LEDGER_KEYS)
    require(ledger["schema"] == LEDGER_SCHEMA
            and ledger["study"] == "sage-mem-v1"
            and ledger["stage"] == "paper-claim-ledger"
            and ledger["status"] == "complete",
            "claim ledger identity/status differs")
    integrity = _mapping(ledger["integrity_completion"],
                         "ledger integrity completion")
    require(integrity.get("status") == "complete"
            and integrity.get("phase_a_cells_verified") == 600
            and integrity.get("finalized_cells_verified") == 600
            and integrity.get("comparators_verified") == 5
            and integrity.get("resources_verified") == 600
            and integrity.get("raw_context_references_verified") == 50,
            "claim ledger integrity counts are incomplete")
    require(integrity.get("phase_a_grid_sha256")
            == report["phase_a_grid_sha256"]
            and integrity.get("identity_ledger_sha256")
            == report["identity_ledger_sha256"],
            "claim ledger formal grid identities differ from the report")

    policy = _mapping(ledger["claim_policy"], "claim policy")
    require(policy.get("per_age_claims_only") is True
            and policy.get("registered_cohorts") == list(COHORTS)
            and policy.get("registered_ages") == list(AGES)
            and policy.get("registered_claim_rows") == 15
            and policy.get("positive_rows_may_not_be_selected_or_omitted")
            is True
            and policy.get("prior_can_substitute_for_host_output") is False
            and policy.get("pooled_cross_host_score_computed") is False
            and policy.get("universal_success_claim_permitted") is False,
            "claim ledger policy crosses a registered boundary")

    claim_rows = ledger["claim_rows"]
    require(isinstance(claim_rows, list) and len(claim_rows) == 15,
            "claim ledger must contain exactly 15 rows")
    expected_pairs = [(cohort, age) for cohort in COHORTS for age in AGES]
    observed_pairs: list[tuple[str, int]] = []
    primary_passes = 0
    execution_passes: list[tuple[str, int]] = []
    for index, row_value in enumerate(claim_rows):
        row = _mapping(row_value, f"claim row {index}")
        cohort = row.get("cohort")
        age = row.get("age")
        require(isinstance(cohort, str)
                and isinstance(age, int) and not isinstance(age, bool),
                f"claim row {index} identity is invalid")
        pair = (cohort, age)
        observed_pairs.append(pair)
        require(pair in report_rows,
                f"claim row {cohort}/age-{age} is not registered")
        require(row.get("cohort_label") == COHORT_LABELS[cohort],
                f"claim row {cohort}/age-{age} label differs")
        report_row = report_rows[pair]
        for field in ROW_REPORT_FIELDS:
            require(field in row and row[field] == report_row.get(field),
                    f"claim row {cohort}/age-{age} field {field} "
                    "differs from the authenticated report")
        report_cohort = report["cohorts"][cohort]
        require(row.get("locked_comparators")
                == report_cohort.get("locked_comparators"),
                f"claim row {cohort}/age-{age} locked comparators differ")
        primary = _bool(row["primary_host_claim_pass"],
                        f"claim row {cohort}/age-{age} primary pass")
        primary_passes += int(primary)
        execution = row.get("execution")
        if execution is not None and execution.get("pass") is True:
            execution_passes.append(pair)
    require(observed_pairs == expected_pairs
            and len(set(observed_pairs)) == 15,
            "claim ledger does not preserve the exact ordered 5 x 3 grid")

    summaries = _mapping(ledger["cohort_summaries"], "cohort summaries")
    require(set(summaries) == set(COHORTS),
            "claim ledger cohort summaries omit or add a cohort")
    summary_keys = {
        "cohort_label", "locked_comparators", "comparator_receipt",
        "backend_admission", "resource_enforcement_verified",
        "registered_age_rows", "rows_passing_primary_host_claim",
        "all_registered_ages_primary_host_claim_pass", "execution_supplied",
        "execution_pass_by_age",
    }
    for cohort in COHORTS:
        summary = _mapping(
            summaries[cohort], f"cohort summary {cohort}", keys=summary_keys)
        report_cohort = report["cohorts"][cohort]
        cohort_rows = [row for row in claim_rows if row["cohort"] == cohort]
        require(summary["cohort_label"] == COHORT_LABELS[cohort]
                and summary["locked_comparators"]
                == report_cohort.get("locked_comparators")
                and summary["comparator_receipt"]
                == report_cohort.get("comparator_receipt")
                and summary["backend_admission"]
                == report_cohort.get("backend_admission")
                and summary["resource_enforcement_verified"] is True
                and summary["registered_age_rows"] == 3
                and summary["rows_passing_primary_host_claim"]
                == sum(row["primary_host_claim_pass"] for row in cohort_rows)
                and summary["all_registered_ages_primary_host_claim_pass"]
                is all(row["primary_host_claim_pass"] for row in cohort_rows)
                and summary["execution_supplied"]
                is report_cohort.get("execution_supplied")
                and summary["execution_pass_by_age"]
                == report_cohort.get("execution_pass_by_age"),
                f"cohort summary {cohort} differs from report/claim rows")

    scientific = _mapping(ledger["scientific_result"], "scientific result")
    require(scientific.get("status") == "evaluated"
            and scientific.get("primary_claim_rows_total") == 15
            and scientific.get("primary_claim_rows_passing") == primary_passes
            and scientific.get("primary_claim_rows_failing")
            == 15 - primary_passes
            and scientific.get("any_primary_claim_row_passed")
            is (primary_passes > 0)
            and scientific.get("all_primary_claim_rows_passed")
            is (primary_passes == 15),
            "scientific-result counts differ from the complete ledger")
    _authenticate_execution_program(ledger["execution_program"], report_rows)
    require(ledger["execution_program"] == report["execution_program"],
            "ledger execution program differs from the authenticated report")
    return {
        "primary_passes": primary_passes,
        "primary_failures": 15 - primary_passes,
        "execution_passes": execution_passes,
    }


def authenticate_source_chain(
        root: Path, ledger: Mapping[str, Any], *, report_path: Path,
        report_payload: bytes, ledger_tex_path: Path,
        ledger_tex_payload: bytes,
        phase_b_binding: Mapping[str, Any]) -> dict[str, Any]:
    source = _mapping(
        ledger["source_binding"], "ledger source binding",
        keys={"report", "protocol", "formal_auditor", "adapter",
              "phase_b_reproduction"})
    report_identity = _mapping(source.get("report"), "bound formal report")
    report_verified = authenticate_identity(
        root, report_identity, "bound formal report",
        expected_path=report_path, require_size=True)
    require(report_identity.get("schema") == REPORT_SCHEMA
            and report_identity.get("expected_sha256_verified") is True
            and report_identity.get(
                "independent_sealed_auditor_recomputation_verified") is True
            and report_verified["sha256"] == sha256_bytes(report_payload),
            "ledger does not bind an independently recomputed formal report")

    protocol = _mapping(source.get("protocol"), "bound protocol")
    protocol_verified = authenticate_identity(
        root, protocol, "bound protocol", require_size=True)
    _digest(protocol.get("fingerprint"), "bound protocol fingerprint")
    lock = authenticate_identity(
        root, protocol.get("implementation_lock"),
        "bound implementation lock", require_size=True)
    amendment = authenticate_identity(
        root, protocol.get("formal_amendment"),
        "bound formal amendment", require_size=True)
    require(protocol["implementation_lock"].get("status") == "sealed",
            "bound implementation lock is not sealed")
    require(protocol["formal_amendment"].get("status")
            == "locked-before-development-selection-or-formal-data",
            "bound formal amendment status differs")
    require(Path(protocol_verified["path"]).name == "sage_mem_v1.yaml"
            and Path(lock["path"]).name == "protocol_lock.json"
            and Path(amendment["path"]).name
            == "sage_mem_v1_formal_amendment.yaml",
            "bound protocol/lock/amendment path identity differs")

    formal_auditor = _mapping(source.get("formal_auditor"),
                              "bound formal auditor")
    auditor_verified = authenticate_identity(
        root, formal_auditor, "bound formal auditor", require_size=True)
    require(formal_auditor.get("sealed_by_implementation_lock") is True
            and Path(auditor_verified["path"]).name
            == "audit_sage_mem_v1_formal.py",
            "formal auditor is not the sealed SAGE-Mem auditor")
    adapter_verified = authenticate_identity(
        root, source.get("adapter"), "bound report adapter")
    require(Path(adapter_verified["path"]).name
            == "summarize_sage_mem_v1_report.py",
            "claim ledger names an unexpected report adapter")
    bound_phase_b = _mapping(
        source.get("phase_b_reproduction"),
        "bound Phase-B reproduction")
    require(dict(bound_phase_b) == dict(phase_b_binding),
            "claim ledger Phase-B receipt/verifier binding differs")
    require(bound_phase_b["operator_pins"]["formal_report_sha256"] ==
            report_verified["sha256"]
            and bound_phase_b["operator_pins"]["protocol_lock_sha256"] ==
            lock["sha256"],
            "claim ledger Phase-B pins do not cross-bind report/protocol")

    publication = _mapping(ledger["publication_artifacts"],
                           "publication artifacts")
    require(set(publication) == {"tex"},
            "publication artifact set differs from the authenticated adapter")
    tex_identity = _mapping(publication.get("tex"), "ledger TeX identity")
    tex_verified = authenticate_identity(
        root, tex_identity, "ledger TeX identity",
        expected_path=ledger_tex_path)
    require(tex_verified["sha256"] == sha256_bytes(ledger_tex_payload),
            "ledger TeX bytes differ from the publication identity")
    return {
        "report": report_verified,
        "protocol": {
            **protocol_verified,
            "fingerprint": protocol["fingerprint"],
        },
        "implementation_lock": lock,
        "formal_amendment": amendment,
        "formal_auditor": auditor_verified,
        "report_adapter": adapter_verified,
        "phase_b_reproduction": dict(bound_phase_b),
        "ledger_tex": tex_verified,
    }


def _tex_escape(value: str) -> str:
    replacements = {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}",
    }
    return "".join(replacements.get(character, character)
                   for character in value)


def _tex_status(value: bool | None) -> str:
    if value is True:
        return r"\SageMemGatePass"
    if value is False:
        return r"\SageMemGateFail"
    return r"\SageMemGateNA"


def _tex_row_list(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return r"\textit{none}"
    return "; ".join(
        f"{_tex_escape(str(row['cohort_label']))} at age {int(row['age'])}"
        for row in rows)


def render_expected_ledger_tex(ledger: Mapping[str, Any]) -> str:
    """Independently reproduce the adapter's complete deterministic TeX."""

    rows = ledger["claim_rows"]
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
        rf"\label{{{LEDGER_SENTINEL}}}%",
        r"\begin{tabular}{llccccccc}",
        r"\hline",
        (r"Cohort & Age & Host & Reset & No state & Controls & Next MSE "
         r"& Primary & Execution \\"),
        r"\hline",
    ]
    for row in rows:
        gates = row["gates"]
        execution = (row["execution"]["pass"]
                     if row["execution"] is not None else None)
        lines.append(
            f"{_tex_escape(row['cohort_label'])} & {row['age']} & "
            f"{_tex_status(gates['host_vs_locked_comparator'])} & "
            f"{_tex_status(gates['full_vs_reset'])} & "
            f"{_tex_status(gates['full_vs_none'])} & "
            f"{_tex_status(gates['all_mechanism_controls'])} & "
            f"{_tex_status(gates['next_mse_noninferiority'])} & "
            f"{_tex_status(row['primary_host_claim_pass'])} & "
            f"{_tex_status(execution)} " + r"\\")
    lines.extend([r"\hline", r"\end{tabular}%", r"}", ""])
    return "\n".join(lines)


def authenticate_ledger_tex(payload: bytes, ledger: Mapping[str, Any]) -> None:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeError as error:
        raise SageMemBindingAuditError(
            "claim-ledger TeX is not UTF-8") from error
    require(text == render_expected_ledger_tex(ledger),
            "claim-ledger TeX is not the exact complete 15-row rendering")


def strip_tex_comments(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        kept: list[str] = []
        for index, character in enumerate(line):
            if character == "%":
                slashes = 0
                cursor = index - 1
                while cursor >= 0 and line[cursor] == "\\":
                    slashes += 1
                    cursor -= 1
                if slashes % 2 == 0:
                    break
            kept.append(character)
        lines.append("".join(kept))
    return "\n".join(lines)


_TEX_EVENT_RE = re.compile(
    r"\\appendix\b|\\(?:input|include)\s*\{([^}]+)\}")
_GRAPHIC_RE = re.compile(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}")


def _resolve_tex_include(root: Path, paper_root: Path, parent: Path,
                         raw: str) -> Path:
    value = Path(raw.strip())
    if not value.suffix:
        value = value.with_suffix(".tex")
    candidates = []
    for base in (parent.parent, paper_root):
        candidate = _lexical_absolute(base / value)
        try:
            candidate.relative_to(_lexical_absolute(root))
        except ValueError:
            continue
        _reject_symlink_components(candidate, label="TeX include")
        if candidate.is_file() and candidate not in candidates:
            candidates.append(candidate)
    require(len(candidates) == 1,
            f"TeX include is missing or ambiguous: {raw} from {parent}")
    require(not candidates[0].is_symlink(),
            f"TeX include is a symlink: {candidates[0]}")
    return candidates[0]


def read_tex_graph(root: Path, main_tex: Path) -> dict[str, Any]:
    root = _lexical_absolute(root)
    main_tex = repository_path(root, main_tex)
    paper_root = main_tex.parent
    fragments: dict[str, list[tuple[Path, str]]] = {
        "main": [], "appendix": []}
    identities: dict[Path, dict[str, Any]] = {}
    include_events: list[dict[str, str]] = []
    active: set[tuple[Path, str]] = set()

    def visit(path: Path, phase: str) -> str:
        key = (path, phase)
        require(key not in active, f"cyclic TeX include: {path}")
        active.add(key)
        payload = read_stable_bytes(path, "paper TeX source")
        try:
            clean = strip_tex_comments(payload.decode("utf-8", errors="strict"))
        except UnicodeError as error:
            raise SageMemBindingAuditError(
                f"paper TeX source is not UTF-8: {path}") from error
        identities[path] = {
            "path": display_path(root, path),
            "size": len(payload),
            "sha256": sha256_bytes(payload),
        }
        cursor = 0
        current = phase
        for match in _TEX_EVENT_RE.finditer(clean):
            fragments[current].append((path, clean[cursor:match.start()]))
            token = match.group(0)
            if token.startswith(r"\appendix"):
                current = "appendix"
            else:
                child = _resolve_tex_include(
                    root, paper_root, path, match.group(1))
                include_events.append({
                    "parent": display_path(root, path),
                    "child": display_path(root, child),
                    "phase": current,
                })
                current = visit(child, current)
            cursor = match.end()
        fragments[current].append((path, clean[cursor:]))
        active.remove(key)
        return current

    visit(main_tex, "main")

    # Local style/class files are executable TeX, so accepting any file found
    # through the TeX search path would defeat the prose audit.  Resolve local
    # package dependencies recursively and require an explicit source pin.
    support_paths: set[Path] = set()
    scanned_support: set[Path] = set()
    while True:
        local_candidates: set[Path] = set()
        for phase in ("main", "appendix"):
            for _source, fragment in fragments[phase]:
                for match in re.finditer(
                        r"\\(?:usepackage|RequirePackage)"
                        r"(?:\[[^]]*\])?\{([^}]+)\}", fragment):
                    for package in match.group(1).split(","):
                        value = Path(package.strip())
                        require(not value.is_absolute()
                                and ".." not in value.parts,
                                "local TeX package name is unsafe")
                        candidate = _lexical_absolute(
                            paper_root / value.with_suffix(".sty"))
                        if candidate.is_file():
                            local_candidates.add(candidate)
                for match in re.finditer(
                        r"\\(?:documentclass|LoadClass)"
                        r"(?:\[[^]]*\])?\{([^}]+)\}", fragment):
                    value = Path(match.group(1).strip())
                    require(not value.is_absolute() and ".." not in value.parts,
                            "local TeX class name is unsafe")
                    candidate = _lexical_absolute(
                        paper_root / value.with_suffix(".cls"))
                    if candidate.is_file():
                        local_candidates.add(candidate)
        pending = local_candidates - scanned_support
        if not pending:
            break
        for path in sorted(pending):
            scanned_support.add(path)
            relative = path.relative_to(root)
            require(relative in TRUSTED_LOCAL_TEX_SUPPORT,
                    f"unregistered local TeX support input: {relative}")
            payload = read_stable_bytes(path, "trusted local TeX support")
            expected = TRUSTED_LOCAL_TEX_SUPPORT[relative]
            require(sha256_bytes(payload) == expected,
                    f"trusted local TeX support hash differs: {relative}")
            try:
                clean = strip_tex_comments(
                    payload.decode("utf-8", errors="strict"))
            except UnicodeError as error:
                raise SageMemBindingAuditError(
                    f"local TeX support is not UTF-8: {relative}") from error
            identities[path] = {
                "path": display_path(root, path),
                "size": len(payload), "sha256": expected,
            }
            fragments["main"].append((path, clean))
            support_paths.add(path)

    # Resolve every static graphic before the build.  Result-bearing SAGE
    # graphics are additionally constrained by their authenticated manifests;
    # all other assets remain bound by this exact pre/post content identity.
    graphic_directories: list[Path] = []
    for phase in ("main", "appendix"):
        for source, fragment in fragments[phase]:
            for declaration in re.finditer(
                    r"\\graphicspath\{((?:\{[^}]+\})+)\}", fragment):
                for raw_directory in re.findall(
                        r"\{([^}]+)\}", declaration.group(1)):
                    for base in (source.parent, paper_root):
                        candidate = _lexical_absolute(base / raw_directory)
                        try:
                            candidate.relative_to(root)
                        except ValueError:
                            continue
                        _reject_symlink_components(
                            candidate, label="graphic search path")
                        if candidate.is_dir() \
                                and candidate not in graphic_directories:
                            graphic_directories.append(candidate)
    graphic_paths: set[Path] = set()
    for phase in ("main", "appendix"):
        for source, fragment in fragments[phase]:
            for match in _GRAPHIC_RE.finditer(fragment):
                graphic = _resolve_graphic(
                    root, paper_root, source, match.group(1),
                    graphic_directories)
                payload = read_stable_bytes(graphic, "paper graphic")
                identities[graphic] = {
                    "path": display_path(root, graphic),
                    "size": len(payload), "sha256": sha256_bytes(payload),
                }
                graphic_paths.add(graphic)
    return {
        "fragments": fragments,
        "identities": [identities[path] for path in sorted(
            identities, key=lambda item: display_path(root, item))],
        "include_events": include_events,
        "reachable_paths": set(identities),
        "tex_reachable_paths": {
            path for path in identities
            if path not in support_paths and path not in graphic_paths},
        "support_paths": support_paths,
        "graphic_paths": graphic_paths,
        "paper_root": paper_root,
    }


def _resolve_graphic(root: Path, paper_root: Path, source: Path,
                     raw: str, search_directories: Sequence[Path] = ()) \
        -> Path:
    value = Path(raw.strip())
    candidates: list[Path] = []
    suffixes = ("",) if value.suffix else (".pdf", ".png", ".jpg", ".jpeg")
    bases = []
    for base in (source.parent, paper_root, *search_directories):
        if base not in bases:
            bases.append(base)
    for base in bases:
        for suffix in suffixes:
            candidate = _lexical_absolute(
                (base / value) if suffix == ""
                else (base / value).with_suffix(suffix))
            try:
                candidate.relative_to(_lexical_absolute(root))
            except ValueError:
                continue
            _reject_symlink_components(candidate, label="graphic")
            if candidate.is_file() and candidate not in candidates:
                candidates.append(candidate)
    require(len(candidates) == 1,
            f"graphic is missing or ambiguous: {raw} from {source}")
    require(not candidates[0].is_symlink(),
            f"graphic is a symlink: {candidates[0]}")
    return candidates[0]


def authenticate_figure_manifests(
        root: Path, manifest_paths: Sequence[Path],
        expected_hashes: Sequence[str], *, report_path: Path,
        report_sha256: str, ledger_path: Path, ledger_sha256: str,
        protocol_fingerprint: str, ledger: Mapping[str, Any]) \
        -> tuple[list[dict[str, Any]], set[Path]]:
    require(len(manifest_paths) == len(expected_hashes),
            "every figure manifest needs one expected SHA-256")
    plotter = _load_claim_plotter() if manifest_paths else None
    receipts: list[dict[str, Any]] = []
    artifacts: set[Path] = set()
    for index, (manifest_path, expected_hash) in enumerate(
            zip(manifest_paths, expected_hashes, strict=True)):
        expected = _digest(expected_hash,
                           f"figure manifest {index} expected SHA-256")
        payload = read_stable_bytes(manifest_path, "SAGE figure manifest")
        require(sha256_bytes(payload) == expected,
                f"figure manifest {index} SHA-256 differs from expected")
        manifest = decode_strict_json(payload, "SAGE figure manifest")
        require(payload == canonical_json(manifest).encode("utf-8"),
                "SAGE figure manifest is not canonical JSON")
        _mapping(manifest, "SAGE figure manifest", keys={
            "schema", "study", "stage", "status", "source_binding",
            "display_contract", "artifacts",
        })
        require(manifest.get("schema") == FIGURE_MANIFEST_SCHEMA
                and manifest.get("study") == "sage-mem-v1"
                and manifest.get("stage") == "publication-plotting"
                and manifest.get("status") == "complete",
                "SAGE figure manifest identity/status differs")
        binding = _mapping(manifest.get("source_binding"),
                           "figure source binding", keys={
                               "claim_ledger", "formal_report_sha256",
                               "protocol_fingerprint", "plotting_script",
                               "phase_b_receipt_sha256",
                               "phase_b_verifier_sha256",
                               "phase_b_registered_contract_sha256",
                           })
        ledger_record = _mapping(
            binding.get("claim_ledger"), "figure-bound claim ledger",
            keys={"path", "size", "sha256", "schema",
                  "expected_sha256_verified"})
        ledger_identity = authenticate_identity(
            root, ledger_record, "figure-bound claim ledger",
            expected_path=ledger_path)
        expected_generator_path = repository_path(root, PLOTTER_SOURCE)
        generator_record = _mapping(
            binding.get("plotting_script"), "figure generator",
            keys={"path", "size", "sha256"})
        generator_identity = authenticate_identity(
            root, generator_record, "figure generator",
            expected_path=expected_generator_path, require_size=True)
        require(plotter is not None, "claim plotter was not loaded")
        runtime_plotter = Path(plotter.__file__)
        runtime_plotter_sha = sha256_file(
            runtime_plotter, "executing claim plotter")
        require(generator_identity["sha256"] == runtime_plotter_sha,
                "figure generator differs from the executing audited plotter")
        require(_digest(binding.get("formal_report_sha256"),
                        "figure-bound report SHA-256") == report_sha256
                and _digest(binding.get("protocol_fingerprint"),
                            "figure-bound protocol fingerprint")
                == protocol_fingerprint
                and ledger_identity["sha256"] == ledger_sha256
                and _digest(binding.get("phase_b_receipt_sha256"),
                            "figure-bound Phase-B receipt SHA-256") ==
                ledger["source_binding"]["phase_b_reproduction"]
                ["receipt"]["sha256"]
                and _digest(binding.get("phase_b_verifier_sha256"),
                            "figure-bound Phase-B verifier SHA-256") ==
                ledger["source_binding"]["phase_b_reproduction"]
                ["verifier"]["sha256"]
                and _digest(
                    binding.get("phase_b_registered_contract_sha256"),
                    "figure-bound Phase-B contract SHA-256") ==
                ledger["source_binding"]["phase_b_reproduction"]
                ["registered_contract_sha256"]
                and ledger_record.get("schema") == LEDGER_SCHEMA
                and ledger_record.get(
                    "expected_sha256_verified") is True,
                "figure manifest is bound to different report/ledger bytes")
        # The report path is deliberately not reopened through the plotting
        # manifest: the renderer only consumes the ledger.  Its report digest
        # must nevertheless equal the separately authenticated report above.
        require(report_path.is_file(), "authenticated formal report vanished")
        expected_display = {
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
                "pass": {"color": plotter.PASS,
                         "shape": "open circle"},
                "fail": {"color": plotter.FAIL, "shape": "cross"},
                "not_evaluated": {"color": plotter.NA,
                                  "shape": "dash"},
            },
        }
        display = _mapping(manifest.get("display_contract"),
                           "figure display contract",
                           keys=set(expected_display))
        require(dict(display) == expected_display,
                "figure display contract differs from the audited renderer")

        try:
            normalized_ledger = plotter.validate_ledger(ledger)
            plot_data = plotter.build_plot_data(
                normalized_ledger, ledger_sha256=ledger_sha256,
                script_sha256=runtime_plotter_sha)
            expected_plot_data = plotter._canonical_json(plot_data)
            claim_pdf, claim_png = plotter.render_claim_ladder(
                plot_data, ledger_sha256=ledger_sha256,
                script_sha256=runtime_plotter_sha)
            effects_pdf, effects_png = plotter.render_effects(
                plot_data, ledger_sha256=ledger_sha256,
                script_sha256=runtime_plotter_sha)
        except plotter.SageMemPlotError as error:
            raise SageMemBindingAuditError(
                "claim ledger cannot be independently re-rendered") from error
        expected_artifacts = {
            "claim_ladder_pdf": claim_pdf,
            "claim_ladder_png": claim_png,
            "effects_pdf": effects_pdf,
            "effects_png": effects_png,
            "plot_data": expected_plot_data,
        }
        records = _mapping(
            manifest.get("artifacts"), "SAGE figure manifest artifacts",
            keys=set(expected_artifacts))
        iterable = list(records.items())
        paper_root = repository_path(root, "paper_a")
        local: list[dict[str, Any]] = []
        for name, record in iterable:
            record = _mapping(
                record, f"figure artifact {name}",
                keys={"path", "size", "sha256"})
            verified = authenticate_identity(
                root, record, f"figure artifact {name}")
            path = repository_path(root, verified["path"])
            try:
                path.relative_to(paper_root)
            except ValueError as error:
                raise SageMemBindingAuditError(
                    f"figure artifact leaves paper_a: {path}") from error
            require(path not in artifacts,
                    f"duplicate SAGE figure artifact: {path}")
            observed = read_stable_bytes(path, f"figure artifact {name}")
            require(observed == expected_artifacts[name],
                    f"figure artifact {name} differs from independent render")
            artifacts.add(path)
            local.append(verified)
        receipts.append({
            "path": display_path(root, manifest_path),
            "size": len(payload),
            "sha256": expected,
            "generator": generator_identity,
            "artifacts": local,
        })
    return receipts, artifacts


def _prose_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^]]*\])?", " ", text)
    normalized = normalized.replace("{", " ").replace("}", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return [item.strip() for item in re.split(r"(?<=[.!?])\s+|\n+", normalized)
            if item.strip()]


_SAGE_MACRO_RE = re.compile(r"\\(SageMem[A-Za-z]+)\b")
_METHOD_ALIAS_RE = re.compile(
    r"\b(?:sage\s*-{0,2}\s*mem|our\s+(?:method|approach|model|carrier)|"
    r"(?:the\s+)?proposed\s+(?:method|approach|model|carrier))\b",
    re.IGNORECASE)
_RESULT_LANGUAGE_RE = re.compile(
    r"\b(?:pass(?:es|ed|ing)?|fail(?:s|ed|ing)?|gain(?:s|ed|ing)?|"
    r"improv\w*|outperform\w*|surpass\w*|accuracy|retention|execution|"
    r"resolved|effect|success|superior|best|win(?:s|ning)?|remember\w*|"
    r"retain\w*|demonstrat\w*|show(?:s|ed|ing)?|achiev\w*|dominat\w*|"
    r"advantage|ahead|lead(?:s|ing)?|beat(?:s|ing)?|better|worse|"
    r"drive(?:s|n)?|enable\w*|plan(?:s|ned|ning)?|recall\w*)\b",
    re.IGNORECASE)
_ROW_LITERAL_RE = re.compile(
    r"\b(?:Reacher|PushT|PointMaze|LeWM|DINO[- ]WM|age\s*[-=]?\s*"
    r"(?:4|8|15))\b", re.IGNORECASE)
_AGE_LITERAL_RE = re.compile(r"\bage\s*[-=]?\s*(?:4|8|15)\b",
                             re.IGNORECASE)
_RAW_RESULT_NUMBER_RE = re.compile(
    r"(?<![A-Za-z])[-+\N{MINUS SIGN}]?\d+(?:\.\d+)?"
    r"\s*[\])}]?\s*(?:pp\b|\\?%|percent\b|"
    r"percentage[- ]points?\b)", re.IGNORECASE)
_FORBIDDEN_SCOPE_RE = re.compile(
    r"\b(?:universal(?:ly)?|every\s+carrier|all\s+carriers|"
    r"pooled(?:\s+cross[- ]host)?|cross[- ]host\s+(?:aggregate|advantage|"
    r"score|gain)|native[- ]planner|native\s+planning|plans?\s+with\s+"
    r"memory|dominat\w*)\b", re.IGNORECASE)
_SAFE_LIMITATION_PATTERNS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    r"^(?:sage\s*-{0,2}\s*mem|our\s+(?:method|approach|model|carrier)|"
    r"(?:the\s+)?proposed\s+(?:method|approach|model|carrier))\s+"
    r"(?:does|do|did)\s+not\s+(?:establish|claim|test|support|demonstrate)"
    r"\b[^.;]*[.]?$",
    r"^(?:we|this\s+paper|this\s+audit|the\s+audit)\s+"
    r"(?:does|do|did)\s+not\s+(?:establish|claim|test|compute|report|"
    r"support|demonstrate)\b[^.;]*[.]?$",
    r"^no\s+[^.;]*(?:claim|score|evidence|comparison|result)\s+"
    r"(?:is|was|has\s+been)\s+(?:made|computed|established|reported|"
    r"claimed)[.]?$",
    r"^(?:this|that|it|sage\s*-{0,2}\s*mem|our\s+(?:method|approach|"
    r"model|carrier)|(?:the\s+)?proposed\s+(?:method|approach|model|"
    r"carrier))\s+(?:is|are)\s+not\s+(?:evidence\s+(?:of|for)\b[^.;]*|"
    r"native\s+planning|a\s+native[- ]planner|a\s+universal(?:ly)?\s+"
    r"superior\s+architecture)[.]?$",
))
_LIMITATION_REVERSAL_RE = re.compile(
    r"\b(?:and|while|although|whereas|but|yet|however|nevertheless|"
    r"nonetheless)\b|;", re.IGNORECASE)


def _honest_limitation(sentence: str) -> bool:
    """Recognize only explicit, non-reversing claim-boundary sentences."""

    normalized = re.sub(r"\s+", " ", sentence.replace("~", " ")).strip()
    return (_LIMITATION_REVERSAL_RE.search(normalized) is None
            and any(pattern.fullmatch(normalized) is not None
                    for pattern in _SAFE_LIMITATION_PATTERNS))


def validate_claim_language(fragments: Mapping[
        str, Sequence[tuple[Path, str]]], *, ledger_tex_path: Path
        ) -> dict[str, Any]:
    """Require all positive SAGE result prose to come from bound macros."""

    macro_counts = {name: 0 for name in sorted(ALLOWED_MANUSCRIPT_MACROS)}
    raw_limitations = 0
    for phase in ("main", "appendix"):
        for path, fragment in fragments[phase]:
            if _lexical_absolute(path) == _lexical_absolute(ledger_tex_path):
                continue
            require(LEDGER_SENTINEL not in fragment,
                    "ledger render sentinel may appear only in the "
                    "authenticated generated ledger")
            require(re.search(
                r"\\(?:newcommand|renewcommand)\s*\{?\\SageMem|"
                r"\\(?:def|let)\s*\\SageMem", fragment) is None,
                f"paper source redefines an authenticated SAGE macro: {path}")
            names = _SAGE_MACRO_RE.findall(fragment)
            unknown = sorted(set(names) - ALLOWED_MANUSCRIPT_MACROS)
            require(not unknown,
                    f"paper invokes unauthenticated SAGE macros: {unknown}")
            for name in names:
                macro_counts[name] += 1

            without_macros = _SAGE_MACRO_RE.sub(" ", fragment)
            for sentence in _prose_sentences(without_macros):
                method_scope = _METHOD_ALIAS_RE.search(sentence) is not None
                row_scope = (_ROW_LITERAL_RE.search(sentence) is not None
                             and _AGE_LITERAL_RE.search(sentence) is not None)
                forbidden_scope = _FORBIDDEN_SCOPE_RE.search(sentence) \
                    is not None
                if _honest_limitation(sentence):
                    if method_scope or row_scope or forbidden_scope:
                        raw_limitations += 1
                    continue
                if ((method_scope or row_scope or forbidden_scope)
                        and (_RESULT_LANGUAGE_RE.search(sentence) is not None
                             or _RAW_RESULT_NUMBER_RE.search(sentence)
                             is not None
                             or forbidden_scope)):
                    raise SageMemBindingAuditError(
                        "positive SAGE result prose/numbers must use only "
                        "authenticated generated macros: "
                        f"{sentence[:180]}")

            # A source-bound macro cannot be used as decoration for an
            # unsupported hard-coded row or number.
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", fragment):
                if _SAGE_MACRO_RE.search(sentence) is None:
                    continue
                residual = _SAGE_MACRO_RE.sub(" ", sentence)
                residual = re.sub(
                    r"\\[A-Za-z@]+\*?(?:\[[^]]*\])?", " ", residual)
                residual = residual.replace("{", " ").replace("}", " ")
                hard_number = _RAW_RESULT_NUMBER_RE.search(residual)
                hard_row_claim = (_ROW_LITERAL_RE.search(residual) is not None
                                  and _RESULT_LANGUAGE_RE.search(residual)
                                  is not None)
                require(hard_number is None and not hard_row_claim,
                        "hard-coded cohort/age result prose cannot accompany "
                        "a generated SAGE macro")
    return {
        "macro_invocations": {
            name: count for name, count in macro_counts.items() if count
        },
        "raw_limitation_sentences": raw_limitations,
    }


def discover_latexmk() -> Path | None:
    """Find latexmk on PATH or in the canonical per-user TinyTeX tree."""

    candidates: list[Path] = []
    on_path = shutil.which("latexmk")
    if on_path:
        candidates.append(Path(on_path))
    tinytex_bin = Path.home() / ".TinyTeX/bin"
    candidates.extend(sorted(tinytex_bin.glob("*/latexmk")))
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return _lexical_absolute(candidate)
    return None


def _generated_build_paths(main_tex: Path) -> dict[str, Path]:
    """Exact build-state allowlist; never glob or remove arbitrary files."""

    stem = main_tex.with_suffix("")
    suffixes = (
        "aux", "out", "toc", "bbl", "bcf", "run.xml", "lof", "lot",
        "fls", "fdb_latexmk", "log", "synctex.gz", "pdf",
    )
    return {suffix: Path(f"{stem}.{suffix}") for suffix in suffixes}


def _trusted_tex_distribution(latexmk: Path) -> tuple[list[Path], list[Path]]:
    """Tie accepted external recorder inputs to the selected TinyTeX tree."""

    lexical = _lexical_absolute(latexmk)
    tinytex = next(
        (parent for parent in (lexical, *lexical.parents)
         if parent.name == ".TinyTeX"), None)
    require(tinytex is not None,
            "selected latexmk is not inside the authenticated TinyTeX tree")
    roots = [tinytex / "texmf-dist", tinytex / "texmf-var"]
    files = [tinytex / "texmf.cnf"]
    for path in (*roots, *files):
        _reject_symlink_components(path, label="trusted TeX distribution")
        require(path.is_dir() if path in roots else path.is_file(),
                f"trusted TeX distribution path is missing: {path}")
    return roots, files


def _sanitized_tex_environment(latexmk: Path) -> tuple[dict[str, str],
                                                        list[Path],
                                                        list[Path]]:
    environment = {
        key: value for key, value in os.environ.items()
        if key not in TEX_SEARCH_ENV_KEYS
        and key.upper() not in TEX_SEARCH_ENV_KEYS
        and not key.upper().startswith("TEXMF")
    }
    environment["PATH"] = (
        f"{latexmk.parent}{os.pathsep}{environment.get('PATH', '')}")
    environment["SOURCE_DATE_EPOCH"] = "946684800"
    environment["FORCE_SOURCE_DATE"] = "1"
    environment["TZ"] = "UTC"
    roots, files = _trusted_tex_distribution(latexmk)
    return environment, roots, files


def run_canonical_paper_build(root: Path, main_tex: Path) -> dict[str, Any]:
    """Force a fresh recorder-enabled canonical paper build."""

    root = _lexical_absolute(root)
    main_tex = repository_path(root, main_tex)
    canonical_main = repository_path(root, CANONICAL_MAIN_TEX)
    require(main_tex == canonical_main,
            "paper build target is not canonical paper_a/main.tex")
    latexmk = discover_latexmk()
    require(latexmk is not None,
            "latexmk/TinyTeX is unavailable for mandatory paper rebuild")
    command = [
        str(latexmk), "-norc", "-g", "-pdf",
        "-pdflatex=pdflatex -no-shell-escape %O %S",
        "-interaction=nonstopmode",
        "-halt-on-error", "-file-line-error", "-recorder", main_tex.name,
    ]
    environment, trusted_roots, trusted_files = \
        _sanitized_tex_environment(latexmk)
    generated = _generated_build_paths(main_tex)
    previous: dict[Path, tuple[bytes, int]] = {}
    for path in generated.values():
        _reject_symlink_components(path, label="paper build-state path")
        require(not path.is_symlink(),
                f"paper build-state path is a symlink: {path}")
        if path.exists():
            require(path.is_file(),
                    f"paper build-state path is not a file: {path}")
            previous[path] = (
                read_stable_bytes(path, "previous paper build state"),
                path.stat().st_mode & 0o777)
    try:
        for path in generated.values():
            if path.exists():
                path.unlink()
        require(not any(path.exists() or path.is_symlink()
                        for path in generated.values()),
                "paper build state was not cleared before compilation")
        directory_fd = os.open(main_tex.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        started_ns = time.time_ns()
        completed = subprocess.run(
            command, cwd=main_tex.parent, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, check=False)
        completed_ns = time.time_ns()
        require(completed.returncode == 0,
                "mandatory canonical paper rebuild failed:\n"
                + completed.stdout[-4000:])
    except BaseException:
        # A failed audit must not destroy the user's last complete PDF/build
        # state.  Restore only the exact enumerated files captured above.
        for path in generated.values():
            if path.exists() and path.is_file() and not path.is_symlink():
                path.unlink()
        for path, (payload, mode) in previous.items():
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".restore", dir=path.parent)
            temporary_path = Path(temporary)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.chmod(temporary_path, mode)
                os.replace(temporary_path, path)
            finally:
                temporary_path.unlink(missing_ok=True)
        directory_fd = os.open(main_tex.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        raise
    engine_source = latexmk.resolve()
    return {
        "engine": str(latexmk),
        "engine_sha256": sha256_bytes(engine_source.read_bytes()),
        "command": command,
        "started_ns": started_ns,
        "completed_ns": completed_ns,
        "returncode": completed.returncode,
        "trusted_external_roots": [str(path) for path in trusted_roots],
        "trusted_external_files": [str(path) for path in trusted_files],
        "tex_search_environment_sanitized": True,
        "shell_escape_disabled": True,
        "fresh_build_state_verified": True,
    }


def _authenticate_build_receipt(value: Any) -> dict[str, Any]:
    record = _mapping(value, "paper build receipt", keys={
        "engine", "engine_sha256", "command", "started_ns",
        "completed_ns", "returncode", "trusted_external_roots",
        "trusted_external_files", "tex_search_environment_sanitized",
        "shell_escape_disabled", "fresh_build_state_verified",
    })
    require(isinstance(record["engine"], str) and record["engine"],
            "paper build engine is missing")
    engine_sha256 = _digest(
        record["engine_sha256"], "paper build engine SHA-256")
    command = record["command"]
    require(isinstance(command, list) and command
            and all(isinstance(item, str) and item for item in command),
            "paper build command is invalid")
    started = _integer(record["started_ns"], "paper build start", minimum=1)
    completed = _integer(
        record["completed_ns"], "paper build completion", minimum=started)
    require(record["returncode"] == 0,
            "paper build runner did not report success")
    roots = record["trusted_external_roots"]
    files = record["trusted_external_files"]
    require(isinstance(roots, list) and isinstance(files, list)
            and all(isinstance(path, str) and path for path in (*roots, *files))
            and record["tex_search_environment_sanitized"] is True
            and record["shell_escape_disabled"] is True
            and record["fresh_build_state_verified"] is True,
            "paper build environment/trust boundary is incomplete")
    return {
        "engine": record["engine"],
        "engine_sha256": engine_sha256,
        "command": list(command),
        "started_ns": started,
        "completed_ns": completed,
        "returncode": 0,
        "trusted_external_roots": list(roots),
        "trusted_external_files": list(files),
        "tex_search_environment_sanitized": True,
        "shell_escape_disabled": True,
        "fresh_build_state_verified": True,
    }


def authenticate_compiled_paper(
        root: Path, graph: Mapping[str, Any], *, main_tex: Path,
        ledger_tex_path: Path, included_figures: set[Path],
        require_ledger_sentinel: bool,
        build_receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Bind a fresh compiled PDF to the audited source graph and ledger."""

    build = _authenticate_build_receipt(build_receipt)
    main_tex = _lexical_absolute(main_tex)
    aux_path = repository_path(root, main_tex.with_suffix(".aux"))
    fls_path = repository_path(root, main_tex.with_suffix(".fls"))
    pdf_path = repository_path(root, main_tex.with_suffix(".pdf"))
    aux = read_stable_bytes(aux_path, "compiled paper AUX")
    fls = read_stable_bytes(fls_path, "compiled paper recorder file")
    pdf = read_stable_bytes(pdf_path, "compiled paper PDF")
    try:
        aux_text = aux.decode("utf-8", errors="strict")
        fls_text = fls.decode("utf-8", errors="strict")
    except UnicodeError as error:
        raise SageMemBindingAuditError(
            "compiled AUX/FLS is not strict UTF-8") from error
    require(pdf.startswith(b"%PDF-") and b"%%EOF" in pdf[-2048:],
            "compiled paper is not a complete PDF")

    pwd_lines = [line[4:].strip() for line in fls_text.splitlines()
                 if line.startswith("PWD ")]
    require(len(pwd_lines) == 1,
            "compiled recorder must contain exactly one PWD")
    compile_pwd = _lexical_absolute(Path(pwd_lines[0]))
    require(compile_pwd == main_tex.parent,
            "compiled recorder PWD is not canonical paper_a")
    recorded_inputs: set[Path] = set()
    for line in fls_text.splitlines():
        if not line.startswith("INPUT "):
            continue
        raw = Path(line[6:].strip())
        path = (_lexical_absolute(raw) if raw.is_absolute()
                else _lexical_absolute(compile_pwd / raw))
        recorded_inputs.add(path)

    root_absolute = _lexical_absolute(root)
    expected_static = {
        _lexical_absolute(path) for path in graph["reachable_paths"]
    }
    # All result graphics must already be members of the static inventory;
    # manifest authentication is an additional, not substitutive, boundary.
    require({_lexical_absolute(path) for path in included_figures}
            .issubset(expected_static),
            "manifest-authenticated result graphic is absent from the "
            "pre-build static inventory")
    generated_paths = _generated_build_paths(main_tex)
    allowed_generated = {
        generated_paths[name] for name in (
            "aux", "out", "toc", "bbl", "bcf", "run.xml", "lof", "lot")
    }
    local_inputs: set[Path] = set()
    external_inputs: set[Path] = set()
    for path in recorded_inputs:
        try:
            path.relative_to(root_absolute)
        except ValueError:
            external_inputs.add(path)
        else:
            local_inputs.add(path)
    observed_generated = local_inputs & allowed_generated
    observed_static = local_inputs - observed_generated
    missing = sorted(display_path(root, path)
                     for path in expected_static - observed_static)
    unexpected = sorted(display_path(root, path)
                        for path in observed_static - expected_static)
    require(not missing and not unexpected,
            "compiled recorder repository-local static input set differs: "
            f"missing={missing}, unexpected={unexpected}")

    prebuild_identities = {
        repository_path(root, record["path"]): record
        for record in graph["identities"]
    }
    static_receipts: list[dict[str, Any]] = []
    for path in sorted(expected_static):
        _reject_symlink_components(path, label="static paper compile input")
        payload = read_stable_bytes(path, "static paper compile input")
        before = prebuild_identities.get(path)
        require(before is not None
                and before["size"] == len(payload)
                and before["sha256"] == sha256_bytes(payload),
                f"static paper input changed during build: {path}")
        role = ("trusted-local-tex-support"
                if path in graph["support_paths"] else
                "graphic" if path in graph["graphic_paths"] else
                "tex-graph")
        static_receipts.append({
            "path": display_path(root, path), "size": len(payload),
            "sha256": sha256_bytes(payload), "role": role,
        })

    generated_receipts: list[dict[str, Any]] = []
    for path in sorted(observed_generated):
        _reject_symlink_components(path, label="generated TeX input")
        payload = read_stable_bytes(path, "generated TeX input")
        require(path.stat().st_mtime_ns >= build["started_ns"],
                f"generated TeX input predates fresh build: {path}")
        generated_receipts.append({
            "path": display_path(root, path), "size": len(payload),
            "sha256": sha256_bytes(payload), "role": "fresh-generated",
        })

    trusted_roots = [_lexical_absolute(path)
                     for path in build["trusted_external_roots"]]
    trusted_files = {_lexical_absolute(path)
                     for path in build["trusted_external_files"]}
    if root_absolute == _lexical_absolute(ROOT):
        engine = _lexical_absolute(build["engine"])
        require(engine.is_file()
                and sha256_file(engine.resolve(), "paper build engine") ==
                build["engine_sha256"],
                "paper build engine identity differs")
        expected_roots, expected_files = _trusted_tex_distribution(engine)
        require(trusted_roots == expected_roots
                and trusted_files == set(expected_files),
                "paper build trusted TeX distribution boundary differs")
    external_receipts: list[dict[str, Any]] = []
    for path in sorted(external_inputs):
        _reject_symlink_components(path, label="external TeX input")
        trusted = path in trusted_files
        if not trusted:
            for trusted_root in trusted_roots:
                try:
                    path.relative_to(trusted_root)
                except ValueError:
                    continue
                trusted = True
                break
        require(trusted,
                f"compiled recorder contains untrusted external input: {path}")
        payload = read_stable_bytes(path, "trusted external TeX input")
        external_receipts.append({
            "path": str(path), "size": len(payload),
            "sha256": sha256_bytes(payload),
            "role": "trusted-tex-distribution",
        })
    if require_ledger_sentinel:
        sentinel = rf"\newlabel{{{LEDGER_SENTINEL}}}"
        require(sentinel in aux_text,
                "compiled AUX does not prove the complete ledger table was "
                "rendered")

    newest_input = max(path.stat().st_mtime_ns for path in expected_static)
    for path in (aux_path, fls_path, pdf_path):
        modified = path.stat().st_mtime_ns
        require(modified >= newest_input
                and modified >= build["started_ns"]
                and modified <= build["completed_ns"] + 1_000_000_000,
                f"compiled paper artifact is stale relative to sources: {path}")

    return {
        "sentinel_required": require_ledger_sentinel,
        "sentinel_verified": (not require_ledger_sentinel
                              or rf"\newlabel{{{LEDGER_SENTINEL}}}"
                              in aux_text),
        "all_audited_inputs_recorded": True,
        "repository_local_static_set_equal": True,
        "untrusted_external_inputs": 0,
        "static_inputs": static_receipts,
        "generated_inputs": generated_receipts,
        "external_inputs": external_receipts,
        "freshness_verified": True,
        "build": {
            "engine": build["engine"],
            "engine_sha256": build["engine_sha256"],
            "command": build["command"],
            "returncode": 0,
            "fresh_rebuild_verified": True,
            "source_date_epoch": 946684800,
            "tex_search_environment_sanitized": True,
            "shell_escape_disabled": True,
            "fresh_build_state_verified": True,
        },
        "artifacts": {
            name: {
                "path": display_path(root, path),
                "size": len(payload),
                "sha256": sha256_bytes(payload),
            }
            for name, path, payload in (
                ("aux", aux_path, aux), ("fls", fls_path, fls),
                ("pdf", pdf_path, pdf))
        },
    }


def inspect_manuscript_integration(
        root: Path, graph: Mapping[str, Any], *, ledger_tex_path: Path,
        figure_artifacts: set[Path], authorization: Mapping[str, Any],
        main_tex: Path,
        compiler_runner: Callable[
            [Path, Path], Mapping[str, Any]] | None = None,
        ) -> tuple[dict[str, Any], dict[str, Any] | None]:
    fragments = graph["fragments"]
    language = validate_claim_language(
        fragments, ledger_tex_path=ledger_tex_path)
    ledger_reachable = (_lexical_absolute(ledger_tex_path)
                        in graph["reachable_paths"])
    ledger_include_phases = {
        event["phase"] for event in graph["include_events"]
        if repository_path(root, event["child"])
        == _lexical_absolute(ledger_tex_path)
    }
    included_figures: set[Path] = set()
    unbound_sage_graphics: list[str] = []
    sage_path_reference = False
    sage_result_language_phase: set[str] = set()
    macro_phase: set[str] = set()
    sage_macro_phase: set[str] = set()
    paper_root = graph["paper_root"]
    graphic_directories: list[Path] = []
    for phase in ("main", "appendix"):
        for source, fragment in fragments[phase]:
            for declaration in re.finditer(
                    r"\\graphicspath\{((?:\{[^}]+\})+)\}", fragment):
                for raw_directory in re.findall(
                    r"\{([^}]+)\}", declaration.group(1)):
                    for base in (source.parent, paper_root):
                        candidate = _lexical_absolute(base / raw_directory)
                        try:
                            candidate.relative_to(_lexical_absolute(root))
                        except ValueError:
                            continue
                        _reject_symlink_components(
                            candidate, label="graphic search path")
                        if candidate.is_dir() \
                                and candidate not in graphic_directories:
                            graphic_directories.append(candidate)
    for phase in ("main", "appendix"):
        for source, fragment in fragments[phase]:
            lowered = fragment.lower()
            if re.search(r"sage[_-]mem[^\s{}]*\.(?:json|tex|pdf|png|jpe?g|svg)",
                         lowered):
                sage_path_reference = True
            for sentence in _prose_sentences(fragment):
                if re.search(r"\bsage[- ]?mem\b", sentence, re.IGNORECASE) \
                        and re.search(
                            r"\b(?:pass|fail|gain|improv|outperform|result|"
                            r"accuracy|retention|execution|resolved|effect)\w*\b",
                            sentence, re.IGNORECASE) \
                        and re.search(
                            r"\b(?:no|not|cannot|without|unresolved)\b",
                            sentence, re.IGNORECASE) is None:
                    sage_result_language_phase.add(phase)
            if _lexical_absolute(source) != _lexical_absolute(ledger_tex_path) \
                    and r"\SageMemClaimLedgerTable" in fragment:
                macro_phase.add(phase)
            if _lexical_absolute(source) != _lexical_absolute(ledger_tex_path) \
                    and re.search(r"\\SageMem[A-Za-z]+", fragment):
                sage_macro_phase.add(phase)
            for match in _GRAPHIC_RE.finditer(fragment):
                graphic = _resolve_graphic(
                    root, paper_root, source, match.group(1),
                    graphic_directories)
                if graphic in figure_artifacts:
                    included_figures.add(graphic)
                else:
                    local = fragment[
                        max(0, match.start() - 320):match.end() + 320]
                    sage_caption = re.search(
                        r"\bsage[- ]?mem\b|\\SageMem", local,
                        re.IGNORECASE) is not None
                    if "sage" in str(graphic.relative_to(
                            _lexical_absolute(root))).lower() or sage_caption:
                        unbound_sage_graphics.append(display_path(root, graphic))
    require(not unbound_sage_graphics,
            "manuscript includes SAGE graphics absent from an authenticated "
            f"manifest: {sorted(unbound_sage_graphics)}")
    integrated = bool(ledger_reachable or included_figures
                      or sage_path_reference or sage_macro_phase
                      or sage_result_language_phase)
    require(not integrated or authorization["publication_authorized"],
            "paper_a references SAGE result artifacts without a registered "
            "positive primary or permitted execution claim")

    mixed_or_negative = authorization["primary_failures"] > 0 \
        or authorization["execution_failures_present"]
    if integrated and mixed_or_negative:
        require(ledger_reachable,
                "mixed/negative SAGE evidence requires the full claim ledger")
        require("appendix" in macro_phase,
                "mixed/negative SAGE evidence requires the complete 15-row "
                "ledger table in the appendix")
    if macro_phase:
        require(ledger_reachable,
                "SAGE ledger macro is used without its authenticated TeX")
    compilation = None
    if integrated:
        build = (compiler_runner or run_canonical_paper_build)(root, main_tex)
        post_build_graph = read_tex_graph(root, main_tex)
        require(post_build_graph == graph,
                "paper TeX source graph changed during the canonical build")
        compilation = authenticate_compiled_paper(
            root, post_build_graph, main_tex=main_tex,
            ledger_tex_path=ledger_tex_path,
            included_figures=included_figures,
            require_ledger_sentinel=mixed_or_negative,
            build_receipt=build)
    return {
        "integrated": integrated,
        "publication_authorized": authorization["publication_authorized"],
        "ledger_reachable": ledger_reachable,
        "ledger_include_phases": sorted(ledger_include_phases),
        "ledger_table_render_phases": sorted(macro_phase),
        "sage_macro_reference_phases": sorted(sage_macro_phase),
        "sage_result_language_phases": sorted(sage_result_language_phase),
        "source_bound_macro_invocations": language["macro_invocations"],
        "honest_raw_limitation_sentences":
            language["raw_limitation_sentences"],
        "authenticated_figures_included": sorted(
            display_path(root, path) for path in included_figures),
        "full_negative_or_mixed_ledger_required":
            bool(integrated and mixed_or_negative),
        "full_negative_or_mixed_ledger_satisfied":
            bool(not integrated or not mixed_or_negative
                 or (ledger_reachable and "appendix" in macro_phase)),
        "forbidden_universal_claims": 0,
        "forbidden_pooled_claims": 0,
        "forbidden_native_planner_claims": 0,
        "compiled_paper_verified": compilation is not None,
    }, compilation


def audit_binding(
        root: Path, *, report: Path, ledger_json: Path, ledger_tex: Path,
        phase_b_receipt: Path, main_tex: Path, expected_report_sha256: str,
        expected_phase_b_receipt_sha256: str,
        expected_ledger_sha256: str,
        figure_manifests: Sequence[Path] = (),
        expected_figure_manifest_sha256: Sequence[str] = (),
        publication_recomputer: Callable[
            [Path, Path, Path, Path, Mapping[str, Any]],
            Mapping[str, bytes]] | None = None,
        phase_b_recomputer: Callable[
            [Path, Mapping[str, Any]], bytes] | None = None,
        compiler_runner: Callable[
            [Path, Path], Mapping[str, Any]] | None = None) \
        -> dict[str, Any]:
    root = _lexical_absolute(root)
    _reject_symlink_components(root, label="repository root")
    if root == _lexical_absolute(ROOT):
        require(publication_recomputer is None,
                "production ROOT cannot override sealed publication "
                "recomputation")
        require(compiler_runner is None,
                "production ROOT cannot override the canonical paper build")
        require(phase_b_recomputer is None,
                "production ROOT cannot override committed Phase-B replay")
    report_path = repository_path(root, report)
    ledger_path = repository_path(root, ledger_json)
    ledger_tex_path = repository_path(root, ledger_tex)
    phase_b_receipt_path = repository_path(root, phase_b_receipt)
    if root == _lexical_absolute(ROOT):
        require(phase_b_receipt_path == repository_path(
                    root, DEFAULT_PHASE_B_RECEIPT),
                "production ROOT requires the canonical Phase-B receipt")
    main_tex_path = repository_path(root, main_tex)
    canonical_main = repository_path(root, CANONICAL_MAIN_TEX)
    require(main_tex_path == canonical_main,
            "--main-tex must be the canonical root/paper_a/main.tex")
    expected_report = _digest(
        expected_report_sha256, "expected report SHA-256")
    expected_ledger = _digest(
        expected_ledger_sha256, "expected claim-ledger SHA-256")

    report_value, report_payload = read_canonical_json(
        report_path, "formal audit report")
    require(sha256_bytes(report_payload) == expected_report,
            "formal report SHA-256 differs from expected")
    report_rows = authenticate_report(report_value)
    phase_b_binding, phase_b_payload = authenticate_phase_b_receipt(
        root, phase_b_receipt_path,
        expected_sha256=expected_phase_b_receipt_sha256,
        report_path=report_path, report_payload=report_payload,
        report_value=report_value,
        nonproduction_test_fixture=(root != _lexical_absolute(ROOT)))
    phase_b_value = decode_strict_json(
        phase_b_payload, "Phase-B reproduction receipt")
    replayed_phase_b = replay_phase_b_receipt(
        root, phase_b_value, recomputer=phase_b_recomputer)
    require(replayed_phase_b == phase_b_payload,
            "committed Phase-B verifier replay differs from canonical "
            "receipt bytes")
    ledger_value, ledger_payload = read_canonical_json(
        ledger_path, "claim-ledger JSON")
    require(sha256_bytes(ledger_payload) == expected_ledger,
            "claim-ledger SHA-256 differs from expected")
    ledger_tex_payload = read_stable_bytes(
        ledger_tex_path, "claim-ledger TeX")
    recompute = publication_recomputer or recompute_authenticated_publication
    recomputed_publication = recompute(
        root, report_path, ledger_path, ledger_tex_path, phase_b_binding)
    authenticate_publication_recomputation(
        recomputed_publication, report_payload=report_payload,
        ledger_payload=ledger_payload,
        ledger_tex_payload=ledger_tex_payload)
    result_counts = authenticate_ledger(
        ledger_value, report_value, report_rows)
    source_chain = authenticate_source_chain(
        root, ledger_value, report_path=report_path,
        report_payload=report_payload, ledger_tex_path=ledger_tex_path,
        ledger_tex_payload=ledger_tex_payload,
        phase_b_binding=phase_b_binding)
    authenticate_ledger_tex(ledger_tex_payload, ledger_value)

    manifest_paths = [repository_path(root, path) for path in figure_manifests]
    manifests, figure_artifacts = authenticate_figure_manifests(
        root, manifest_paths, expected_figure_manifest_sha256,
        report_path=report_path, report_sha256=expected_report,
        ledger_path=ledger_path, ledger_sha256=expected_ledger,
        protocol_fingerprint=ledger_value["source_binding"]["protocol"][
            "fingerprint"], ledger=ledger_value)
    graph = read_tex_graph(root, main_tex_path)

    program = ledger_value["execution_program"]
    permitted_execution = []
    execution_failures_present = False
    for row in ledger_value["claim_rows"]:
        execution = row["execution"]
        if execution is not None and execution["pass"] is False:
            execution_failures_present = True
        if execution is not None and execution["pass"] is True \
                and program["per_age"][str(row["age"])][
                    "claim_permitted"] is True:
            permitted_execution.append((row["cohort"], row["age"]))
    authorization = {
        "primary_positive_rows": result_counts["primary_passes"],
        "primary_failures": result_counts["primary_failures"],
        "permitted_execution_positive_rows": len(permitted_execution),
        "execution_failures_present": execution_failures_present,
        "publication_authorized": bool(
            result_counts["primary_passes"] or permitted_execution),
    }
    integration, compilation = inspect_manuscript_integration(
        root, graph, ledger_tex_path=ledger_tex_path,
        figure_artifacts=figure_artifacts, authorization=authorization,
        main_tex=main_tex_path, compiler_runner=compiler_runner)
    auditor_path = Path(__file__).resolve()
    return {
        "schema": SCHEMA,
        "study": "sage-mem-v1",
        "status": "verified",
        "read_only_campaign_audit": True,
        "experiment_artifacts_read": True,
        "formal_outcomes_recomputed": True,
        "sealed_publication_chain_recomputed": True,
        "phase_b_receipt_replayed": True,
        "phase_b_recomputer_injected": phase_b_recomputer is not None,
        "publication_recomputer_injected":
            publication_recomputer is not None,
        "compiler_runner_injected": compiler_runner is not None,
        "coverage": {
            "registered_cohorts": list(COHORTS),
            "registered_ages": list(AGES),
            "claim_rows_verified": 15,
            "exact_order_verified": True,
            "positive_row_selection_permitted": False,
        },
        "authorization": authorization,
        "integration": integration,
        "identities": {
            "formal_report": {
                "path": display_path(root, report_path),
                "size": len(report_payload),
                "sha256": expected_report,
            },
            "claim_ledger_json": {
                "path": display_path(root, ledger_path),
                "size": len(ledger_payload),
                "sha256": expected_ledger,
            },
            "phase_b_reproduction": phase_b_binding,
            "source_chain": source_chain,
            "figure_manifests": manifests,
            "paper_sources": graph["identities"],
            "compiled_paper": compilation,
            "auditor": {
                "path": display_path(root, auditor_path),
                "size": auditor_path.stat().st_size,
                "sha256": sha256_file(auditor_path, "binding auditor"),
            },
        },
    }


def emit_receipt(root: Path, destination: Path, payload: Mapping[str, Any],
                 *, execute: bool, resume: bool = False) -> str:
    if not execute:
        return "not-written"
    require(payload.get("schema") == SCHEMA
            and payload.get("status") == "verified",
            "refusing to write an unverified binding receipt")
    root = _lexical_absolute(root)
    _reject_symlink_components(root, label="repository root")
    target = repository_path(root, destination)
    expected = canonical_json(payload).encode("utf-8")
    if target.exists() or target.is_symlink():
        require(resume, f"refusing to overwrite existing receipt: {target}")
        require(target.is_file() and not target.is_symlink(),
                f"resume receipt is unsafe: {target}")
        require(read_stable_bytes(target, "existing binding receipt")
                == expected,
                "resume receipt differs from the current authenticated audit")
        return "validated-existing"
    target.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(target.parent, label="receipt destination")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(expected)
            stream.flush()
            os.fsync(stream.fileno())
        require(not target.exists() and not target.is_symlink(),
                f"receipt appeared concurrently: {target}")
        try:
            os.link(temporary_path, target)
        except FileExistsError as error:
            raise SageMemBindingAuditError(
                f"receipt appeared concurrently: {target}") from error
        parent_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        temporary_path.unlink(missing_ok=True)
    return "created"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--ledger-json", "--claim-ledger-json",
                        dest="ledger_json", type=Path,
                        default=DEFAULT_LEDGER_JSON)
    parser.add_argument("--ledger-tex", "--claim-ledger-tex",
                        dest="ledger_tex", type=Path,
                        default=DEFAULT_LEDGER_TEX)
    parser.add_argument("--phase-b-receipt", type=Path,
                        default=DEFAULT_PHASE_B_RECEIPT)
    parser.add_argument("--main-tex", type=Path, default=DEFAULT_MAIN_TEX)
    parser.add_argument("--figure-manifest", type=Path, action="append")
    parser.add_argument("--expected-report-sha256")
    parser.add_argument("--expected-phase-b-receipt-sha256")
    parser.add_argument("--expected-ledger-sha256",
                        "--expected-claim-ledger-sha256",
                        dest="expected_ledger_sha256")
    parser.add_argument("--expected-figure-manifest-sha256", action="append")
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _preview(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "study": "sage-mem-v1",
        "preview": True,
        "report": str(args.report),
        "ledger_json": str(args.ledger_json),
        "ledger_tex": str(args.ledger_tex),
        "phase_b_receipt": str(args.phase_b_receipt),
        "main_tex": str(args.main_tex),
        "figure_manifests": [str(path)
                             for path in (args.figure_manifest or [])],
        "receipt": str(args.receipt),
        "required_claim_rows": 15,
        "required_cohorts": list(COHORTS),
        "required_ages": list(AGES),
        "no_files_read": True,
        "no_files_written": True,
        "no_outcomes_read": True,
    }


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.execute:
        if args.resume:
            print(canonical_json({
                "schema": SCHEMA, "status": "failed",
                "reason": "--resume requires --execute",
                "receipt_written": False,
            }), end="")
            return 1
        print(canonical_json(_preview(args)), end="")
        return 0
    try:
        require(args.expected_report_sha256 is not None,
                "--expected-report-sha256 is required with --execute")
        require(args.expected_ledger_sha256 is not None,
                "--expected-ledger-sha256 is required with --execute")
        require(args.expected_phase_b_receipt_sha256 is not None,
                "--expected-phase-b-receipt-sha256 is required with "
                "--execute")
        manifests = tuple(args.figure_manifest or ())
        manifest_hashes = tuple(
            args.expected_figure_manifest_sha256 or ())
        payload = audit_binding(
            args.root, report=args.report, ledger_json=args.ledger_json,
            ledger_tex=args.ledger_tex,
            phase_b_receipt=args.phase_b_receipt, main_tex=args.main_tex,
            expected_report_sha256=args.expected_report_sha256,
            expected_phase_b_receipt_sha256=
            args.expected_phase_b_receipt_sha256,
            expected_ledger_sha256=args.expected_ledger_sha256,
            figure_manifests=manifests,
            expected_figure_manifest_sha256=manifest_hashes)
        publication = emit_receipt(
            args.root, args.receipt, payload, execute=True,
            resume=bool(args.resume))
        output = dict(payload)
        output["receipt"] = {
            "path": str(args.receipt), "publication": publication,
        }
        print(canonical_json(output), end="")
        return 0
    except SageMemBindingAuditError as error:
        print(canonical_json({
            "schema": SCHEMA,
            "study": "sage-mem-v1",
            "status": "failed",
            "receipt_written": False,
            "reason": str(error),
        }), end="")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
