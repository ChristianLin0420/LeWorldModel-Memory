#!/usr/bin/env python3
"""Read-only confirmatory evidence audit for finalized SAGE-Mem v1.

The auditor treats the frozen-host ``full`` read as the primary memory
endpoint.  The carrier ``prior`` is retained as a diagnostic and can never
rescue a failed host-output claim.  Every contrast is reported separately at
evidence ages 4, 8, and 15; no pooled cross-host score is computed.

Inputs are the immutable 600-cell Phase-A grid, the post-reveal finalized
grid, and the formal-preparation receipts that bind the locked development
comparators.  This module launches no model, controller, or formal job.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_SPEC = ROOT / "configs/sage_mem_v1.yaml"
AUDIT_SCHEMA = "sage_mem_v1_formal_evidence_audit_v1"
FINALIZER_SCHEMA = "sage_mem_v1_formal_finalizer_v1"
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
SEEDS = tuple(range(10))
AGES = (4, 8, 15)
BASELINE_ARMS = (
    "gru", "lstm", "ssm", "fixed_trust", "gdelta",
    "fixed_trust_aux", "ssm_aux",
)
TRAINABLE_ARMS = tuple(arm for arm in ARMS if arm != "none")
MECHANISM_CONTROLS = (
    "sage_mem_next_only", "sage_mem_no_exposure",
    "sage_mem_exposure_only", "fixed_trust_aux", "ssm_aux",
)
RESOURCE_FIELDS = (
    "trainable_parameters", "forward_flops_per_episode",
    "persistent_state_floats", "peak_cuda_bytes",
    "wall_clock_train_seconds",
)
BASE_FINAL_ARRAYS = {
    "formal_test_episode_id",
    "formal_test_native_cluster_id",
    "formal_test_evidence_age",
    "formal_test_label",
    "formal_test_full_pred",
    "formal_test_reset_pred",
    "formal_test_prior_pred",
    "formal_test_full_correct",
    "formal_test_reset_correct",
    "formal_test_prior_correct",
    "formal_test_full_mse",
    "formal_test_reset_mse",
    "formal_test_prior_mse",
}
EXECUTION_ARRAYS = {
    "formal_test_full_execution_success",
    "formal_test_reset_execution_success",
    "formal_test_prior_execution_success",
}
RAW_ARRAYS = {
    "formal_test_episode_id",
    "formal_test_native_cluster_id",
    "formal_test_evidence_age",
    "formal_test_label",
    "formal_test_short_pred",
    "formal_test_long_pred",
    "formal_test_short_correct",
    "formal_test_long_correct",
}


class FormalEvidenceAuditError(RuntimeError):
    """Formal evidence is incomplete, unpaired, or violates a locked gate."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FormalEvidenceAuditError(message)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _read_json(path: Path, label: str) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise FormalEvidenceAuditError(
            f"non-finite JSON constant in {label}: {value}")
    try:
        result = json.loads(path.read_text(encoding="utf-8"),
                            parse_constant=reject)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FormalEvidenceAuditError(
            f"cannot read {label}: {path}") from error
    _require(isinstance(result, dict), f"{label} is not a JSON mapping")
    return result


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(_canonical_json(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FormalEvidenceAuditError(
                f"refusing to overwrite existing audit report: {path}") \
                from error
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _safe_artifact(parent: Path, record: Mapping[str, Any], label: str) -> Path:
    _require(set(record) == {"path", "sha256", "size"},
             f"{label} artifact handle schema changed")
    relative = Path(str(record.get("path", "")))
    _require(not relative.is_absolute() and ".." not in relative.parts
             and len(relative.parts) == 1,
             f"{label} artifact path is unsafe")
    path = parent / relative
    _require(path.is_file() and not path.is_symlink()
             and path.stat().st_size == record.get("size")
             and _is_sha256(record.get("sha256"))
             and _sha256_file(path) == record["sha256"],
             f"{label} artifact identity failed")
    return path


@dataclass(frozen=True)
class AuditContract:
    cohorts: tuple[str, ...]
    arms: tuple[str, ...]
    seeds: tuple[int, ...]
    ages: tuple[int, ...]
    classes: Mapping[str, int]
    formal_rows: Mapping[str, int]
    variants: Mapping[str, int]
    target_parameters: Mapping[str, int]
    bootstrap_draws: int
    bootstrap_seed: int
    parameter_margin: float
    flop_margin: float
    host_comparator_gain: float
    reset_gain: float
    reset_mse_ratio_max: float
    mechanism_gain: float
    mse_relative_margin: float
    execution_gain: float
    execution_oracle_gate: float
    require_raw_context: bool = True
    require_600_cells: bool = True

    @property
    def total_cells(self) -> int:
        return len(self.cohorts) * len(self.arms) * len(self.seeds)


def contract_from_spec(spec: Mapping[str, Any], *,
                       require_raw_context: bool = True) -> AuditContract:
    gates = spec["confirmatory_gates"]
    return AuditContract(
        cohorts=COHORTS,
        arms=ARMS,
        seeds=SEEDS,
        ages=AGES,
        classes={cohort: int(spec["cohorts"][cohort]["classes"])
                 for cohort in COHORTS},
        formal_rows={
            cohort: int(spec["cohorts"][cohort][
                "split_episodes"]["formal_test"])
            * (4 if cohort == "dinowm_pointmaze_goal" else 1)
            for cohort in COHORTS},
        variants={cohort: (4 if cohort == "dinowm_pointmaze_goal" else 1)
                  for cohort in COHORTS},
        target_parameters={
            cohort: int(spec["cohorts"][cohort]["target_parameters"])
            for cohort in COHORTS},
        bootstrap_draws=int(spec["statistics"]["bootstrap_draws"]),
        bootstrap_seed=int(spec["statistics"]["bootstrap_seed"]),
        parameter_margin=float(spec["fairness_reporting"][
            "maximum_parameter_relative_gap"]),
        flop_margin=float(spec["fairness_reporting"][
            "maximum_flop_relative_gap"]),
        host_comparator_gain=float(gates["host_output_exposure"][
            "minimum_absolute_gain"]),
        reset_gain=float(gates["reset_causality"][
            "minimum_absolute_drop"]),
        reset_mse_ratio_max=float(gates["reset_causality"][
            "reset_to_full_mse_ratio_max"]),
        mechanism_gain=float(gates["mechanism_controls"][
            "minimum_absolute_gain"]),
        mse_relative_margin=float(gates["next_feature_noninferiority"][
            "relative_margin"]),
        execution_gain=float(gates["execution"][
            "minimum_absolute_success_gain"]),
        execution_oracle_gate=float(gates["execution"]["oracle_gate"]),
        require_raw_context=require_raw_context,
        require_600_cells=True,
    )


@dataclass(frozen=True)
class LoadedFormalEvidence:
    contract: AuditContract
    phase_a_grid_sha256: str
    cells: Mapping[tuple[str, str, int], Mapping[str, np.ndarray]]
    resources: Mapping[tuple[str, str, int], Mapping[str, int | float]]
    comparators: Mapping[str, Mapping[str, str]]
    comparator_receipts: Mapping[str, Mapping[str, Any]]
    backend_admissions: Mapping[str, Mapping[str, Any]]
    raw_context: Mapping[tuple[str, int], Mapping[str, np.ndarray]]
    execution_status: Mapping[str, Mapping[str, Any]]
    execution_random: Mapping[str, np.ndarray]
    phase_a_cells_verified: int
    finalized_cells_verified: int
    raw_context_references_verified: int
    identity_ledger_sha256: str


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: np.asarray(archive[name]).copy()
                    for name in archive.files}
    except (OSError, ValueError) as error:
        raise FormalEvidenceAuditError(f"cannot load NPZ: {path}") from error


def _validate_cluster_multiplicity(
        episode: np.ndarray, cluster: np.ndarray, labels: np.ndarray,
        *, variants: int, classes: int, label: str) -> None:
    _require(episode.ndim == cluster.ndim == labels.ndim == 1
             and len(episode) == len(cluster) == len(labels),
             f"{label} identity arrays are unaligned")
    _require(len(np.unique(episode)) == len(episode),
             f"{label} episode IDs are not unique")
    unique, counts = np.unique(cluster, return_counts=True)
    _require(len(unique) * variants == len(cluster)
             and np.all(counts == variants),
             f"{label} does not preserve x{variants} native clusters")
    if variants == classes:
        for identity in unique:
            _require(set(map(int, labels[cluster == identity]))
                     == set(range(classes)),
                     f"{label} counterfactual cluster omits a class")


def _validate_finalized_arrays(
        arrays: Mapping[str, np.ndarray], *, cohort: str,
        contract: AuditContract,
        phase_identity: Mapping[str, np.ndarray]) -> None:
    keys = set(arrays)
    _require(keys in (BASE_FINAL_ARRAYS,
                      BASE_FINAL_ARRAYS | EXECUTION_ARRAYS),
             f"finalized result schema differs: {cohort}")
    expected_shape = (len(contract.ages), contract.formal_rows[cohort])
    values = {key: np.asarray(value) for key, value in arrays.items()}
    _require(all(value.shape == expected_shape for value in values.values()),
             f"finalized arrays are not age-by-row: {cohort}")
    for key in ("formal_test_episode_id",
                "formal_test_native_cluster_id",
                "formal_test_evidence_age"):
        _require(np.array_equal(values[key], phase_identity[key]),
                 f"finalized identity differs from Phase A: {cohort}/{key}")
    labels = values["formal_test_label"]
    _require(np.issubdtype(labels.dtype, np.integer)
             and np.all((labels >= 0) &
                        (labels < contract.classes[cohort]))
             and all(np.array_equal(labels[index], labels[0])
                     for index in range(1, len(contract.ages))),
             f"finalized labels are invalid or drift by age: {cohort}")
    _validate_cluster_multiplicity(
        values["formal_test_episode_id"][0],
        values["formal_test_native_cluster_id"][0], labels[0],
        variants=contract.variants[cohort],
        classes=contract.classes[cohort], label=f"{cohort}/formal_test")
    for stream in ("full", "reset", "prior"):
        pred = values[f"formal_test_{stream}_pred"]
        correct = values[f"formal_test_{stream}_correct"]
        mse = values[f"formal_test_{stream}_mse"]
        _require(np.issubdtype(pred.dtype, np.integer)
                 and np.all((pred >= 0) &
                            (pred < contract.classes[cohort])),
                 f"finalized prediction is out of range: {cohort}/{stream}")
        _require(np.array_equal(correct,
                                (pred == labels).astype(correct.dtype))
                 and np.isin(correct, (0, 1)).all(),
                 f"finalized correctness disagrees with prediction: "
                 f"{cohort}/{stream}")
        _require(np.issubdtype(mse.dtype, np.number)
                 and np.isfinite(mse).all() and np.all(mse >= 0),
                 f"finalized MSE is invalid: {cohort}/{stream}")
    for key in EXECUTION_ARRAYS.intersection(values):
        _require(np.isin(values[key], (0, 1)).all(),
                 f"execution success is not binary: {cohort}/{key}")


def _load_resource_report(cell: Any) -> tuple[dict[str, int | float], str]:
    manifest = _read_json(
        cell.directory / "manifest.json", "Phase-A cell manifest")
    record = manifest.get("artifacts", {}).get("resource_report")
    _require(isinstance(record, Mapping),
             "Phase-A resource artifact handle is missing")
    path = _safe_artifact(cell.directory, record, "Phase-A resource")
    value = _read_json(path, "Phase-A resource report")
    _require(set(value) == {"schema", "study", "status", "metrics"}
             and value.get("schema") == "sage_mem_v1_phase_a_resources_v1"
             and value.get("study") == "sage-mem-v1"
             and value.get("status") == "complete"
             and isinstance(value.get("metrics"), Mapping)
             and set(value["metrics"]) == set(RESOURCE_FIELDS),
             "Phase-A resource report schema changed")
    metrics = dict(value["metrics"])
    _require(all(isinstance(metrics[key], (int, float))
                 and not isinstance(metrics[key], bool)
                 and math.isfinite(float(metrics[key]))
                 and float(metrics[key]) >= 0 for key in RESOURCE_FIELDS),
             "Phase-A resource metric is invalid")
    return metrics, str(record["sha256"])


def _authenticated_identity(
        record: Any, *, root: Path, label: str,
        expected_path: Path | None = None) -> Path:
    _require(isinstance(record, Mapping)
             and set(record) == {"path", "size", "sha256"}
             and isinstance(record.get("size"), int)
             and record["size"] >= 0 and _is_sha256(record.get("sha256")),
             f"{label} identity is malformed")
    declared = Path(str(record["path"]))
    if declared.is_absolute():
        path = declared.resolve()
    else:
        _require(".." not in declared.parts,
                 f"{label} identity path is unsafe")
        path = (root / declared).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as error:
            raise FormalEvidenceAuditError(
                f"{label} identity leaves preparation root") from error
    _require(path.is_file() and not path.is_symlink()
             and path.stat().st_size == record["size"]
             and _sha256_file(path) == record["sha256"]
             and (expected_path is None
                  or path == expected_path.resolve()),
             f"{label} identity changed")
    return path


def _backend_admission(
        cohort: str, proof: Any, bank_manifest: Mapping[str, Any],
        bank_manifest_sha256: str,
        ) -> dict[str, Any]:
    _require(isinstance(proof, Mapping),
             f"backend admission proof is malformed: {cohort}")
    if cohort.startswith("lewm_"):
        admissions = bank_manifest.get("admissions")
        required = {
            "backend", "host_hash_before", "host_hash_after",
            "parent_overlap_zero", "formal_split_overlap_zero",
        }
        _require(set(proof) == required
                 and proof.get("backend") == "SIGReg-LeWM"
                 and _is_sha256(proof.get("host_hash_before"))
                 and proof.get("host_hash_before") ==
                 proof.get("host_hash_after")
                 and proof.get("parent_overlap_zero") is True
                 and proof.get("formal_split_overlap_zero") is True
                 and bank_manifest.get("host_hash_before") ==
                 proof["host_hash_before"]
                 and bank_manifest.get("host_hash_after") ==
                 proof["host_hash_after"]
                 and isinstance(admissions, Mapping)
                 and admissions.get(
                     "parent_overlap_zero") is True
                 and admissions.get(
                     "formal_split_overlap_zero") is True,
                 f"LeWM backend admission/freshness proof failed: {cohort}")
    else:
        freshness = bank_manifest.get("freshness_proof")
        required = {
            "backend", "plan_sha256", "provenance_manifest_sha256",
            "parent_episode_overlap_count",
            "cross_split_native_episode_overlap_count",
        }
        _require(set(proof) == required
                 and proof.get("backend") == "DINO-WM"
                 and _is_sha256(proof.get("plan_sha256"))
                 and _is_sha256(proof.get("provenance_manifest_sha256"))
                 and proof.get("provenance_manifest_sha256") ==
                 bank_manifest_sha256
                 and proof.get("parent_episode_overlap_count") == 0
                 and proof.get(
                     "cross_split_native_episode_overlap_count") == 0
                 and bank_manifest.get("plan_sha256") ==
                 proof["plan_sha256"]
                 and isinstance(freshness, Mapping)
                 and freshness.get(
                     "parent_episode_overlap_count") == 0
                 and freshness.get(
                     "cross_split_native_episode_overlap_count") == 0,
                 f"DINO backend admission/freshness proof failed: {cohort}")
    return {**dict(proof), "admission_rechecked": True}


def _load_locked_comparators(
        prepare_root: Path, grid: Any, contract: AuditContract,
        spec: Mapping[str, Any]) -> tuple[
            dict[str, Mapping[str, str]],
            dict[str, Mapping[str, Any]],
            dict[str, Mapping[str, Any]], Path]:
    from scripts.sage_mem_v1_spec import spec_fingerprint

    preparation_manifest = _read_json(
        prepare_root / "manifest.json", "formal preparation manifest")
    preparation_manifest_sha256 = _sha256_file(prepare_root / "manifest.json")
    required_manifest = {
        "schema", "study", "status", "formal_jobs_launched",
        "formal_outcomes_read", "development_outcomes_read",
        "development_access", "study_protocol", "implementation_lock",
        "custody_registry", "cohort_receipts",
    }
    _require(set(preparation_manifest) == required_manifest
             and preparation_manifest.get("schema") ==
             "sage_mem_v1_formal_preparation_v1"
             and preparation_manifest.get("study") == "sage-mem-v1"
             and preparation_manifest.get("status") == "complete"
             and preparation_manifest.get("formal_jobs_launched") is False
             and preparation_manifest.get("formal_outcomes_read") is False
             and preparation_manifest.get("development_outcomes_read")
             is False
             and preparation_manifest.get("development_access") ==
             "opaque locked comparator receipt identities only",
             "formal preparation manifest boundary changed")
    spec_path = Path(str(spec.get("_spec_path", ""))).resolve()
    spec_sha256 = str(spec.get("_spec_sha256", ""))
    _require(_is_sha256(spec_sha256), "loaded protocol SHA-256 is missing")
    protocol_path = _authenticated_identity(
        preparation_manifest["study_protocol"], root=prepare_root,
        label="formal preparation protocol", expected_path=spec_path)
    _require(_sha256_file(protocol_path) == spec_sha256,
             "formal preparation protocol digest differs")
    lock_value = spec.get("implementation_lock")
    _require(isinstance(lock_value, str) and lock_value,
             "registered implementation lock is missing")
    lock_path = _authenticated_identity(
        preparation_manifest["implementation_lock"], root=prepare_root,
        label="formal preparation implementation lock",
        expected_path=ROOT / lock_value)
    registry_path = _authenticated_identity(
        preparation_manifest["custody_registry"], root=prepare_root,
        label="formal preparation custody registry")
    registry = _read_json(registry_path, "formal preparation custody registry")
    _require(registry.get("schema") ==
             "sage_mem_v1_custody_vault_registry_v1"
             and registry.get("study") == "sage-mem-v1"
             and registry.get("status") == "sealed"
             and registry.get(
                 "labels_available_only_after_complete_phase_a_grid") is True
             and registry.get("development_outcomes_read") is False
             and isinstance(registry.get("cohorts"), Mapping)
             and set(registry["cohorts"]) == set(contract.cohorts),
             "formal preparation custody registry boundary changed")
    cohort_identities = preparation_manifest["cohort_receipts"]
    _require(isinstance(cohort_identities, Mapping)
             and set(cohort_identities) == set(contract.cohorts),
             "formal preparation cohort receipt registry changed")
    comparators: dict[str, Mapping[str, str]] = {}
    receipts: dict[str, Mapping[str, Any]] = {}
    admissions: dict[str, Mapping[str, Any]] = {}
    for cohort in contract.cohorts:
        path = _authenticated_identity(
            cohort_identities[cohort], root=prepare_root,
            label=f"{cohort} preparation receipt",
            expected_path=prepare_root / "receipts" / f"{cohort}.json")
        value = _read_json(path, f"{cohort} preparation receipt")
        required = {
            "schema", "study", "status", "cohort",
            "formal_jobs_launched", "formal_outcomes_read",
            "development_outcomes_read", "development_access", "boundaries",
            "bank_manifest", "custody_receipt", "sealed_vault_sha256",
            "backend_proof", "finalizer_custody_record",
        }
        _require(set(value) == required
                 and value.get("schema") ==
                 "sage_mem_v1_formal_prepared_cohort_v1"
                 and value.get("study") == "sage-mem-v1"
                 and value.get("status") == "prepared-and-hash-validated"
                 and value.get("cohort") == cohort
                 and value.get("formal_jobs_launched") is False
                 and value.get("formal_outcomes_read") is False
                 and value.get("development_outcomes_read") is False
                 and value.get("development_access") ==
                 "opaque locked comparator receipt identity only",
                 f"formal preparation receipt changed: {cohort}")
        bank = value["bank_manifest"]
        bank_path = _authenticated_identity(
            bank, root=prepare_root,
            label=f"{cohort} prepared bank manifest")
        _require(bank.get("sha256") == grid.bank_hashes[cohort],
                 f"preparation bank differs from Phase A: {cohort}")
        bank_manifest = _read_json(
            bank_path, f"{cohort} prepared bank manifest")
        custody_receipt_path = _authenticated_identity(
            value["custody_receipt"], root=prepare_root,
            label=f"{cohort} custody receipt")
        custody_receipt = _read_json(
            custody_receipt_path, f"{cohort} custody receipt")
        custody_sources = value.get("finalizer_custody_record", {}).get(
            "sources", {})
        source_hashes = {
            source.get("artifact", {}).get("sha256")
            for source in custody_sources.values()
            if isinstance(source, Mapping)
        } if isinstance(custody_sources, Mapping) else set()
        _require(_is_sha256(value.get("sealed_vault_sha256"))
                 and custody_receipt.get("sha256") ==
                 value["sealed_vault_sha256"]
                 and source_hashes == {value["sealed_vault_sha256"]}
                 and value.get("finalizer_custody_record") ==
                 registry["cohorts"][cohort],
                 f"preparation custody binding differs: {cohort}")
        boundaries = value.get("boundaries")
        _require(isinstance(boundaries, Mapping)
                 and set(boundaries) == {
                     "study_protocol", "implementation_lock",
                     "locked_comparator_receipt"}
                 and boundaries.get("study_protocol") ==
                 preparation_manifest["study_protocol"]
                 and boundaries.get("implementation_lock") ==
                 preparation_manifest["implementation_lock"],
                 f"preparation protocol differs: {cohort}")
        _authenticated_identity(
            boundaries["study_protocol"], root=prepare_root,
            label=f"{cohort} preparation protocol",
            expected_path=protocol_path)
        _authenticated_identity(
            boundaries["implementation_lock"], root=prepare_root,
            label=f"{cohort} implementation lock", expected_path=lock_path)
        identity = boundaries.get("locked_comparator_receipt")
        selection_path = _authenticated_identity(
            identity, root=prepare_root,
            label=f"{cohort} locked comparator receipt")
        selection = _read_json(
            selection_path, f"{cohort} locked comparator receipt")
        locked = selection.get("locked_comparators")
        _require(selection.get("schema_version") == 1
                 and selection.get("study") == "sage-mem-v1"
                 and selection.get("stage") == "development-selection"
                 and selection.get("status") == "selected"
                 and selection.get("cohort") == cohort
                 and selection.get("protocol_fingerprint") ==
                 spec_fingerprint(spec)
                 and isinstance(locked, Mapping)
                 and set(locked) == {"retention", "next_feature", "execution"}
                 and all(arm in BASELINE_ARMS for arm in locked.values())
                 and isinstance(selection.get("gdelta_development_healthy"),
                                bool)
                 and selection.get(
                     "labels_used_only_for_posthoc_selection_metrics") is True
                 and selection.get("formal_data_read") is False
                 and selection.get("formal_execution_started") is False,
                 f"locked comparator receipt schema changed: {cohort}")
        _require(selection.get("gdelta_development_healthy") is True
                 or "gdelta" not in locked.values(),
                 f"unhealthy gDelta is a locked comparator: {cohort}")
        comparators[cohort] = dict(locked)
        receipts[cohort] = {
            "formal_preparation_manifest_sha256":
                preparation_manifest_sha256,
            "implementation_lock_sha256":
                preparation_manifest["implementation_lock"]["sha256"],
            "custody_registry_sha256":
                preparation_manifest["custody_registry"]["sha256"],
            "preparation_receipt": dict(cohort_identities[cohort]),
            "locked_comparator_receipt": {
                "path": str(selection_path.resolve()),
                "size": selection_path.stat().st_size,
                "sha256": identity["sha256"],
            },
        }
        admissions[cohort] = _backend_admission(
            cohort, value["backend_proof"], bank_manifest,
            str(bank["sha256"]))
    return comparators, receipts, admissions, registry_path


def _load_finalized_grid(
        finalized_root: Path, grid: Any, contract: AuditContract
        ) -> tuple[dict[tuple[str, str, int], Mapping[str, np.ndarray]],
                   dict[tuple[str, int], Mapping[str, np.ndarray]],
                   dict[str, Mapping[str, Any]], str]:
    summary_path = finalized_root / "summary.json"
    summary = _read_json(summary_path, "formal finalizer summary")
    required_summary = {
        "schema", "study", "stage", "status", "phase_a_cells",
        "phase_a_grid_sha256", "label_reveal_receipt_sha256",
        "label_registry_sha256", "development_outcomes_read",
        "per_age_results_preserved", "pointmaze_x4_native_clustering_preserved",
        "raw_context_reference", "execution_decks",
        "finalized_cells_sha256", "finalized_cells",
    }
    _require(set(summary) == required_summary
             and summary.get("schema") == FINALIZER_SCHEMA
             and summary.get("study") == "sage-mem-v1"
             and summary.get("stage") == "formal-finalizer"
             and summary.get("status") == "complete"
             and summary.get("phase_a_cells") == contract.total_cells
             and summary.get("finalized_cells") == contract.total_cells
             and summary.get("phase_a_grid_sha256") == grid.grid_sha256
             and summary.get("development_outcomes_read") is False
             and summary.get("per_age_results_preserved") is True
             and summary.get("pointmaze_x4_native_clustering_preserved")
             == (contract.variants.get("dinowm_pointmaze_goal") == 4),
             "formal finalizer summary identity changed")
    reveal_path = finalized_root / "label_reveal_receipt.json"
    _require(reveal_path.is_file() and not reveal_path.is_symlink()
             and _is_sha256(summary.get("label_reveal_receipt_sha256"))
             and _sha256_file(reveal_path) ==
             summary["label_reveal_receipt_sha256"]
             and _is_sha256(summary.get("label_registry_sha256")),
             "formal reveal/label identity is missing or changed")
    reveal = _read_json(reveal_path, "formal label-reveal receipt")
    required_reveal = {
        "schema", "study", "stage", "status",
        "complete_grid_validated_before_label_reveal",
        "formal_test_labels_read_before_receipt", "development_outcomes_read",
        "phase_a_cells", "phase_a_grid_sha256", "bank_manifest_sha256",
        "raw_context_reference", "execution_deck_registry",
        "label_registry", "recorded_unix_ns",
    }
    registry_identity = reveal.get("label_registry")
    _require(set(reveal) == required_reveal
             and reveal.get("schema") == FINALIZER_SCHEMA
             and reveal.get("study") == "sage-mem-v1"
             and reveal.get("stage") == "label-reveal"
             and reveal.get("status") ==
             "authorized-after-complete-phase-a-grid"
             and reveal.get("complete_grid_validated_before_label_reveal")
             is True
             and reveal.get("formal_test_labels_read_before_receipt") is False
             and reveal.get("development_outcomes_read") is False
             and reveal.get("phase_a_cells") == contract.total_cells
             and reveal.get("phase_a_grid_sha256") == grid.grid_sha256
             and reveal.get("bank_manifest_sha256") == dict(grid.bank_hashes)
             and isinstance(registry_identity, Mapping)
             and set(registry_identity) == {"path", "sha256", "size"}
             and registry_identity.get("sha256") ==
             summary["label_registry_sha256"],
             "formal label-reveal receipt is stale or malformed")
    raw_receipt = reveal["raw_context_reference"]
    _require(isinstance(raw_receipt, Mapping)
             and set(raw_receipt) == {
                 "status", "sha256", "separate_from_parameter_matched_arms",
                 "short_context_frames", "long_context_frames"}
             and raw_receipt.get("status") == (
                 "validated" if contract.require_raw_context
                 else raw_receipt.get("status"))
             and (not contract.require_raw_context
                  or _is_sha256(raw_receipt.get("sha256")))
             and raw_receipt.get(
                 "separate_from_parameter_matched_arms") is True
             and raw_receipt.get("short_context_frames") == 3
             and raw_receipt.get("long_context_frames") == 16,
             "raw-context reveal identity is malformed")
    registry_path = Path(str(registry_identity["path"]))
    _require(registry_path.is_file() and not registry_path.is_symlink()
             and registry_path.stat().st_size == registry_identity["size"]
             and _sha256_file(registry_path) == registry_identity["sha256"],
             "sealed label registry changed after reveal")

    execution_summary = summary["execution_decks"]
    required_execution_summary = {
        "status", "supplied_cohorts", "eligible_cohorts",
        "program_requires_at_least_two_eligible_cohorts",
        "program_gate_passed", "cohort_status",
    }
    _require(isinstance(execution_summary, Mapping)
             and set(execution_summary) == required_execution_summary
             and isinstance(execution_summary.get("cohort_status"), Mapping),
             "formal execution summary is malformed")
    execution_status = dict(execution_summary["cohort_status"])
    supplied = execution_summary.get("supplied_cohorts")
    _require(isinstance(supplied, list)
             and supplied == sorted(execution_status)
             and set(supplied).issubset(contract.cohorts)
             and execution_summary.get("status") ==
             ("evaluated" if supplied else "not-supplied"),
             "formal execution cohort registry differs")
    eligible_count = 0
    for cohort, status in execution_status.items():
        required_status = {
            "status", "eligible", "receipt_sha256",
            "oracle_success", "random_success",
        }
        _require(isinstance(status, Mapping)
                 and set(status) == required_status
                 and isinstance(status.get("eligible"), bool)
                 and _is_sha256(status.get("receipt_sha256"))
                 and isinstance(status.get("oracle_success"), (int, float))
                 and not isinstance(status.get("oracle_success"), bool)
                 and isinstance(status.get("random_success"), (int, float))
                 and not isinstance(status.get("random_success"), bool)
                 and math.isfinite(float(status["oracle_success"]))
                 and math.isfinite(float(status["random_success"]))
                 and 0.0 <= float(status["oracle_success"]) <= 1.0
                 and 0.0 <= float(status["random_success"]) <= 1.0,
                 f"execution status is malformed: {cohort}")
        eligible = status["eligible"] is True
        eligible_count += int(eligible)
        _require(status["status"] == (
            "computed-class-conditioned-arm-blind" if eligible
            else "skipped-oracle-gate")
            and ((float(status["oracle_success"]) >=
                  contract.execution_oracle_gate) == eligible),
            f"execution oracle gate differs: {cohort}")
        receipt_path = finalized_root / "execution" / cohort / "receipt.json"
        _require(receipt_path.is_file() and not receipt_path.is_symlink()
                 and _sha256_file(receipt_path) == status["receipt_sha256"],
                 f"execution receipt identity changed: {cohort}")
        receipt = _read_json(receipt_path, f"{cohort} execution receipt")
        _require(receipt.get("schema") == FINALIZER_SCHEMA
                 and receipt.get("study") == "sage-mem-v1"
                 and receipt.get("stage") == "external-execution"
                 and receipt.get("cohort") == cohort
                 and receipt.get("status") == status["status"]
                 and receipt.get("eligible") is eligible
                 and receipt.get("controller_pinned") is True
                 and receipt.get("arm_identity_used") is False
                 and receipt.get("input") == "predicted_class_only"
                 and receipt.get("oracle_success") ==
                 status["oracle_success"]
                 and receipt.get("random_success") ==
                 status["random_success"]
                 and receipt.get("eligibility_threshold") ==
                 contract.execution_oracle_gate,
                 f"execution receipt content changed: {cohort}")
    program_required = execution_summary.get(
        "program_requires_at_least_two_eligible_cohorts")
    _require(execution_summary.get("eligible_cohorts") == eligible_count
             and isinstance(program_required, bool)
             and execution_summary.get("program_gate_passed") ==
             (eligible_count >= 2 if program_required else None)
             and (not program_required or eligible_count >= 2),
             "formal execution program gate differs")
    if contract.require_600_cells:
        _require(contract.total_cells == 600,
                 "production formal audit must contain exactly 600 cells")
    cells_root = finalized_root / "cells"
    expected_directories = {
        (cells_root / cohort / arm / f"seed-{seed}").resolve()
        for cohort in contract.cohorts for arm in contract.arms
        for seed in contract.seeds
    }
    observed_directories = {
        path.parent.resolve() for path in cells_root.glob("*/*/seed-*/manifest.json")
    }
    _require(observed_directories == expected_directories,
             "finalized 600-cell directory registry is incomplete or has extras")
    cells: dict[tuple[str, str, int], Mapping[str, np.ndarray]] = {}
    records: list[dict[str, Any]] = []
    reference_labels: dict[str, np.ndarray] = {}
    consumer_hashes: dict[tuple[str, int], str] = {}
    for cohort in contract.cohorts:
        for arm in contract.arms:
            for seed in contract.seeds:
                key = (cohort, arm, seed)
                directory = cells_root / cohort / arm / f"seed-{seed}"
                manifest_path = directory / "manifest.json"
                manifest = _read_json(manifest_path, "finalized cell manifest")
                required = {
                    "schema", "study", "stage", "status", "cohort", "arm",
                    "seed", "ages", "phase_a_grid_sha256",
                    "phase_a_manifest_sha256", "bank_manifest_sha256",
                    "label_registry_sha256", "label_reveal_receipt_sha256",
                    "shared_arm_blind_consumer_sha256",
                    "native_cluster_id_preserved",
                    "counterfactual_variants_per_native_cluster",
                    "execution", "artifact",
                }
                cell = grid.cells[key]
                _require(set(manifest) == required
                         and manifest.get("schema") == FINALIZER_SCHEMA
                         and manifest.get("study") == "sage-mem-v1"
                         and manifest.get("stage") == "formal-finalized"
                         and manifest.get("status") == "complete"
                         and manifest.get("cohort") == cohort
                         and manifest.get("arm") == arm
                         and manifest.get("seed") == seed
                         and manifest.get("ages") == list(contract.ages)
                         and manifest.get("phase_a_grid_sha256") == grid.grid_sha256
                         and manifest.get("phase_a_manifest_sha256")
                         == cell.manifest_sha256
                         and manifest.get("bank_manifest_sha256")
                         == cell.bank_sha256
                         and manifest.get("label_registry_sha256") ==
                         summary["label_registry_sha256"]
                         and manifest.get("label_reveal_receipt_sha256") ==
                         summary["label_reveal_receipt_sha256"]
                         and manifest.get("native_cluster_id_preserved") is True
                         and manifest.get(
                             "counterfactual_variants_per_native_cluster")
                         == contract.variants[cohort],
                         f"finalized cell identity changed: {key}")
                artifact = _safe_artifact(
                    directory, manifest["artifact"], "finalized result")
                arrays = _load_npz(artifact)
                phase_identity = {
                    name: cell.arrays[name] for name in (
                        "formal_test_episode_id",
                        "formal_test_native_cluster_id",
                        "formal_test_evidence_age")}
                _validate_finalized_arrays(
                    arrays, cohort=cohort, contract=contract,
                    phase_identity=phase_identity)
                if cohort not in reference_labels:
                    reference_labels[cohort] = arrays["formal_test_label"]
                else:
                    _require(np.array_equal(
                        reference_labels[cohort],
                        arrays["formal_test_label"]),
                        f"finalized labels drift across cells: {cohort}")
                shared = manifest.get("shared_arm_blind_consumer_sha256")
                _require(_is_sha256(shared),
                         f"shared consumer identity is malformed: {key}")
                consumer_key = (cohort, seed)
                if consumer_key in consumer_hashes:
                    _require(consumer_hashes[consumer_key] == shared,
                             f"consumer differs across arms: {consumer_key}")
                else:
                    consumer_hashes[consumer_key] = str(shared)
                execution = manifest.get("execution")
                status = execution_status.get(cohort)
                expected_execution = status is not None and \
                    status["eligible"] is True
                expected_status = (
                    "computed-class-conditioned-arm-blind"
                    if expected_execution else
                    ("skipped-oracle-gate" if status is not None
                     else "not-supplied"))
                _require(isinstance(execution, Mapping)
                         and set(execution) == {
                             "status", "eligible", "receipt_sha256",
                             "arm_identity_used", "ages", "per_age_success"}
                         and execution.get("arm_identity_used") is False
                         and execution.get("ages") == list(contract.ages)
                         and execution.get("status") == expected_status
                         and execution.get("eligible") == (
                             True if expected_execution else
                             (False if status is not None else None))
                         and execution.get("receipt_sha256") == (
                             status["receipt_sha256"]
                             if status is not None else None),
                         f"execution manifest is malformed: {key}")
                has_execution = EXECUTION_ARRAYS.issubset(arrays)
                _require(has_execution == expected_execution,
                         f"execution arrays/gate disagree: {key}")
                cells[key] = arrays
                records.append({
                    "cohort": cohort, "arm": arm, "seed": seed,
                    "artifact_sha256": manifest["artifact"]["sha256"],
                    "consumer_sha256": shared,
                })
    _require(summary.get("finalized_cells_sha256") == _sha256_json(records),
             "finalized cell ledger hash differs")

    raw_summary = summary["raw_context_reference"]
    _require(isinstance(raw_summary, Mapping),
             "raw-context summary is malformed")
    raw: dict[tuple[str, int], Mapping[str, np.ndarray]] = {}
    raw_records: list[dict[str, Any]] = []
    raw_source_records: list[dict[str, Any]] = []
    if contract.require_raw_context:
        _require(raw_summary.get("status") == "complete"
                 and raw_summary.get("short_context_frames") == 3
                 and raw_summary.get("long_context_frames") == 16
                 and raw_summary.get(
                     "separate_from_parameter_matched_arms") is True
                 and raw_summary.get("references") ==
                 len(contract.cohorts) * len(contract.seeds),
                 "required short-3/long-16 reference is incomplete")
        for cohort in contract.cohorts:
            canonical = cells[(cohort, contract.arms[0], contract.seeds[0])]
            for seed in contract.seeds:
                directory = finalized_root / "raw_context" / cohort / \
                    f"seed-{seed}"
                manifest = _read_json(
                    directory / "manifest.json", "raw-context manifest")
                required = {
                    "schema", "study", "stage", "status", "cohort", "seed",
                    "ages", "short_context_frames", "long_context_frames",
                    "separate_from_parameter_matched_arms",
                    "phase_a_grid_sha256", "source_manifest_sha256",
                    "shared_arm_blind_consumer_sha256", "artifact",
                }
                _require(set(manifest) == required
                         and manifest.get("schema") == FINALIZER_SCHEMA
                         and manifest.get("stage") ==
                         "formal-raw-context-finalized"
                         and manifest.get("status") == "complete"
                         and manifest.get("cohort") == cohort
                         and manifest.get("seed") == seed
                         and manifest.get("ages") == list(contract.ages)
                         and manifest.get("short_context_frames") == 3
                         and manifest.get("long_context_frames") == 16
                         and manifest.get(
                             "separate_from_parameter_matched_arms") is True
                         and manifest.get("phase_a_grid_sha256") ==
                         grid.grid_sha256
                         and _is_sha256(
                             manifest.get("source_manifest_sha256"))
                         and _is_sha256(manifest.get(
                             "shared_arm_blind_consumer_sha256")),
                         f"raw-context finalized identity changed: "
                         f"{cohort}/seed-{seed}")
                artifact = _safe_artifact(
                    directory, manifest["artifact"], "raw-context result")
                consumer = _read_json(
                    finalized_root / "raw_context_consumers" / cohort /
                    f"seed-{seed}.json", "raw-context consumer receipt")
                required_consumer = {
                    "schema", "study", "stage", "status", "cohort", "seed",
                    "contexts", "ages_pooled", "training_rows",
                    "feature_dimension", "arm_identity_used",
                    "context_identity_used", "formal_test_labels_used",
                    "model_sha256", "shared_consumer_sha256",
                }
                _require(set(consumer) == required_consumer
                         and consumer.get("schema") == FINALIZER_SCHEMA
                         and consumer.get("stage") ==
                         "raw-context-shared-consumer"
                         and consumer.get("status") ==
                         "fit-after-complete-grid-reveal"
                         and consumer.get("cohort") == cohort
                         and consumer.get("seed") == seed
                         and consumer.get("contexts") ==
                         ["short-3", "long-16"]
                         and consumer.get("ages_pooled") ==
                         list(contract.ages)
                         and isinstance(consumer.get("feature_dimension"), int)
                         and consumer["feature_dimension"] > 0
                         and consumer.get("arm_identity_used") is False
                         and consumer.get("context_identity_used") is False
                         and consumer.get("formal_test_labels_used") is False
                         and _is_sha256(consumer.get("model_sha256"))
                         and consumer.get("shared_consumer_sha256") ==
                         manifest["shared_arm_blind_consumer_sha256"],
                         f"raw-context consumer identity changed: "
                         f"{cohort}/seed-{seed}")
                arrays = _load_npz(artifact)
                _require(set(arrays) == RAW_ARRAYS,
                         "raw-context finalized schema changed")
                expected_shape = (
                    len(contract.ages), contract.formal_rows[cohort])
                _require(all(value.shape == expected_shape
                             for value in arrays.values()),
                         "raw-context arrays are not age-by-row")
                for name in ("formal_test_episode_id",
                             "formal_test_native_cluster_id",
                             "formal_test_evidence_age",
                             "formal_test_label"):
                    _require(np.array_equal(arrays[name], canonical[name]),
                             f"raw-context pairing differs: {cohort}/{name}")
                for stream in ("short", "long"):
                    pred = arrays[f"formal_test_{stream}_pred"]
                    correct = arrays[f"formal_test_{stream}_correct"]
                    _require(np.issubdtype(pred.dtype, np.integer)
                        and np.all((pred >= 0) &
                                   (pred < contract.classes[cohort]))
                        and np.array_equal(
                        correct,
                        (pred == arrays["formal_test_label"]).astype(
                            correct.dtype))
                        and np.isin(correct, (0, 1)).all(),
                        f"raw-context metric is invalid: {cohort}/{stream}")
                raw[(cohort, seed)] = arrays
                raw_records.append({
                    "cohort": cohort, "seed": seed,
                    "artifact_sha256": manifest["artifact"]["sha256"],
                    "consumer_sha256": manifest[
                        "shared_arm_blind_consumer_sha256"],
                })
                raw_source_records.append({
                    "cohort": cohort, "seed": seed,
                    "manifest_sha256": manifest[
                        "source_manifest_sha256"],
                    "bank_manifest_sha256": grid.bank_hashes[cohort],
                    "feature_dimension": consumer["feature_dimension"],
                })
        _require(raw_summary.get("records_sha256") ==
                 _sha256_json(raw_records),
                 "raw-context finalized ledger hash differs")
        _require(raw_receipt.get("sha256") ==
                 _sha256_json(raw_source_records),
                 "raw-context source ledger differs from reveal receipt")
    else:
        _require(raw_summary.get("status") in {"complete", "not-supplied"},
                 "raw-context summary status changed")
    return cells, raw, dict(execution_status), str(summary["phase_a_grid_sha256"])


def _infer_execution_registry(finalized_root: Path) -> Path | None:
    reveal = _read_json(
        finalized_root / "label_reveal_receipt.json",
        "formal label-reveal receipt")
    record = reveal.get("execution_deck_registry")
    _require(isinstance(record, Mapping)
             and set(record) == {"status", "path", "sha256", "size"},
             "execution-deck reveal identity is malformed")
    if record.get("status") == "not-supplied":
        _require(record == {
            "status": "not-supplied", "path": None,
            "sha256": None, "size": None},
            "absent execution-deck reveal identity changed")
        return None
    _require(record.get("status") == "sealed-supplied"
             and _is_sha256(record.get("sha256"))
             and isinstance(record.get("size"), int),
             "sealed execution-deck reveal identity is malformed")
    path = Path(str(record["path"])).resolve()
    _require(path.is_file() and not path.is_symlink()
             and path.stat().st_size == record["size"]
             and _sha256_file(path) == record["sha256"],
             "sealed execution-deck registry identity changed")
    return path


def load_formal_evidence(
        *, spec: Mapping[str, Any], phase_a_root: str | Path,
        finalized_root: str | Path, prepare_root: str | Path,
        raw_context_root: str | Path | None = None,
        contract: AuditContract | None = None) -> LoadedFormalEvidence:
    """Authenticate Phase A, finalized cells, resources, and comparators."""

    from scripts.sage_mem_v1_formal_finalizer import (
        SageMemFormalFinalizerError, _load_execution_decks,
        _load_label_registry,
        validate_complete_phase_a_grid, validate_finalized_output,
    )

    contract = contract or contract_from_spec(spec)
    _require(tuple(contract.cohorts) == COHORTS
             and tuple(contract.arms) == ARMS
             and tuple(contract.seeds) == SEEDS
             and tuple(contract.ages) == AGES,
             "production loader requires the registered 5x12x10x3 contract")
    grid = validate_complete_phase_a_grid(phase_a_root)
    _require(len(grid.cells) == contract.total_cells == 600,
             "Phase-A validation did not authenticate exactly 600 cells")
    prepare_path = Path(prepare_root).resolve()
    finalized_path = Path(finalized_root).resolve()
    comparators, comparator_receipts, backend_admissions, registry_path = \
        _load_locked_comparators(prepare_path, grid, contract, spec)
    inferred_raw = (Path(raw_context_root).resolve()
                    if raw_context_root is not None
                    else prepare_path.parent / "raw_context_phase_a")
    _require(inferred_raw.is_dir() and not inferred_raw.is_symlink(),
             "registered raw_context_phase_a root is missing or unsafe")
    execution_registry = _infer_execution_registry(finalized_path)
    try:
        validate_finalized_output(
            phase_a_root, registry_path, finalized_path,
            raw_context_root=inferred_raw,
            execution_deck_registry=execution_registry)
    except SageMemFormalFinalizerError as error:
        raise FormalEvidenceAuditError(
            f"official finalized-output validation failed: {error}") from error
    reveal = _read_json(
        finalized_path / "label_reveal_receipt.json",
        "formal label-reveal receipt")
    raw_digest = reveal["raw_context_reference"]["sha256"]
    try:
        if execution_registry is not None:
            labels = _load_label_registry(
                registry_path,
                finalized_path / "label_reveal_receipt.json", grid,
                raw_context_sha256=raw_digest,
                execution_deck_registry=execution_registry)
            decks = dict(_load_execution_decks(
                execution_registry, registry_path,
                finalized_path / "label_reveal_receipt.json", grid, labels,
                raw_context_sha256=raw_digest))
        else:
            decks = {}
    except SageMemFormalFinalizerError as error:
        raise FormalEvidenceAuditError(
            f"execution-deck authentication failed: {error}") from error
    execution_random = {
        cohort: np.asarray(deck.random_success, dtype=np.uint8).copy()
        for cohort, deck in decks.items()
    }
    resources: dict[tuple[str, str, int], Mapping[str, int | float]] = {}
    resource_hashes: list[dict[str, Any]] = []
    for key, cell in grid.cells.items():
        metrics, digest = _load_resource_report(cell)
        resources[key] = metrics
        resource_hashes.append({"cohort": key[0], "arm": key[1],
                                "seed": key[2], "sha256": digest})
    cells, raw, execution, phase_hash = _load_finalized_grid(
        finalized_path, grid, contract)
    _require(phase_hash == grid.grid_sha256,
             "finalized output differs from Phase-A grid")
    ledger = _sha256_json({
        "phase_a_grid_sha256": grid.grid_sha256,
        "resource_artifacts": resource_hashes,
        "comparator_receipts": comparator_receipts,
        "finalized_cells": len(cells),
        "raw_context_references": len(raw),
    })
    return LoadedFormalEvidence(
        contract=contract,
        phase_a_grid_sha256=grid.grid_sha256,
        cells=cells,
        resources=resources,
        comparators=comparators,
        comparator_receipts=comparator_receipts,
        backend_admissions=backend_admissions,
        raw_context=raw,
        execution_status=execution,
        execution_random=execution_random,
        phase_a_cells_verified=len(grid.cells),
        finalized_cells_verified=len(cells),
        raw_context_references_verified=len(raw),
        identity_ledger_sha256=ledger,
    )


def _cluster_profiles(labels: np.ndarray, clusters: np.ndarray
                      ) -> list[np.ndarray]:
    unique = np.unique(clusters)
    rows = {identity: np.flatnonzero(clusters == identity)
            for identity in unique}
    profiles: dict[tuple[int, ...], list[int]] = {}
    for identity in unique:
        profile = tuple(sorted(map(int, labels[rows[identity]])))
        profiles.setdefault(profile, []).append(int(identity))
    result = []
    for identities in profiles.values():
        result.append(np.asarray([
            rows[identity] for identity in identities], dtype=object))
    return result


def _profile_cluster_matrix(
        values: np.ndarray, cluster_rows: np.ndarray) -> np.ndarray:
    result = np.empty((values.shape[0], len(cluster_rows)), dtype=np.float64)
    for index, rows in enumerate(cluster_rows):
        result[:, index] = values[:, np.asarray(rows, dtype=np.int64)].mean(
            axis=1)
    return result


def paired_seed_cluster_bootstrap(
        left: np.ndarray, right: np.ndarray, labels: np.ndarray,
        clusters: np.ndarray, *, draws: int, seed: int,
        confidence: float = 0.95, statistic: str = "difference",
        chunk_size: int = 256) -> dict[str, Any]:
    """Crossed paired seed×native-cluster percentile bootstrap."""

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    clusters = np.asarray(clusters, dtype=np.int64)
    _require(left.shape == right.shape and left.ndim == 2
             and left.shape[1] == len(labels) == len(clusters)
             and left.size > 0 and np.isfinite(left).all()
             and np.isfinite(right).all(),
             "paired bootstrap inputs are malformed")
    _require(statistic in {"difference", "relative_ratio"},
             "unknown paired bootstrap statistic")
    _require(isinstance(draws, int) and draws > 0
             and 0 < confidence < 1,
             "bootstrap draw/confidence contract is invalid")
    profiles = _cluster_profiles(labels, clusters)
    _require(profiles, "bootstrap has no native clusters")
    left_matrices = [_profile_cluster_matrix(left, rows) for rows in profiles]
    right_matrices = [_profile_cluster_matrix(right, rows) for rows in profiles]

    def combine(left_values: Sequence[float],
                right_values: Sequence[float]) -> float:
        left_mean = float(np.mean(left_values))
        right_mean = float(np.mean(right_values))
        if statistic == "difference":
            return left_mean - right_mean
        if right_mean == 0.0:
            return 0.0 if left_mean == 0.0 else float("inf")
        return left_mean / right_mean - 1.0

    point = combine([matrix.mean() for matrix in left_matrices],
                    [matrix.mean() for matrix in right_matrices])
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    seed_count = left.shape[0]
    uniform_seed = np.full(seed_count, 1.0 / seed_count)
    for offset in range(0, draws, chunk_size):
        stop = min(draws, offset + chunk_size)
        count = stop - offset
        seed_weights = rng.multinomial(
            seed_count, uniform_seed, size=count).astype(np.float64)
        sampled_left = np.zeros((count, len(profiles)), dtype=np.float64)
        sampled_right = np.zeros_like(sampled_left)
        for profile_index, (left_matrix, right_matrix) in enumerate(zip(
                left_matrices, right_matrices, strict=True)):
            clusters_count = left_matrix.shape[1]
            cluster_weights = rng.multinomial(
                clusters_count,
                np.full(clusters_count, 1.0 / clusters_count),
                size=count).astype(np.float64)
            denominator = float(seed_count * clusters_count)
            sampled_left[:, profile_index] = np.einsum(
                "bi,ij,bj->b", seed_weights, left_matrix, cluster_weights,
                optimize=True) / denominator
            sampled_right[:, profile_index] = np.einsum(
                "bi,ij,bj->b", seed_weights, right_matrix, cluster_weights,
                optimize=True) / denominator
        left_mean = sampled_left.mean(axis=1)
        right_mean = sampled_right.mean(axis=1)
        if statistic == "difference":
            samples[offset:stop] = left_mean - right_mean
        else:
            zero = right_mean == 0
            samples[offset:stop] = np.divide(
                left_mean, right_mean,
                out=np.full(count, np.inf), where=~zero) - 1.0
            samples[offset:stop][zero & (left_mean == 0)] = 0.0
    _require(np.isfinite(samples).all(),
             "relative bootstrap encountered a zero reference")
    alpha = (1.0 - confidence) / 2.0
    return {
        "point": float(point),
        "lower": float(np.quantile(samples, alpha)),
        "upper": float(np.quantile(samples, 1.0 - alpha)),
        "confidence": float(confidence),
        "draws": int(draws),
        "seed": int(seed),
        "resampling_unit": "paired formal seed x native episode cluster",
        "class_profile_stratified": True,
        "pairing_preserved": True,
    }


def _metric_matrix(
        evidence: LoadedFormalEvidence, cohort: str, arm: str,
        key: str, age_index: int) -> np.ndarray:
    return np.stack([
        np.asarray(evidence.cells[(cohort, arm, seed)][key][age_index],
                   dtype=np.float64)
        for seed in evidence.contract.seeds
    ])


def _raw_matrix(evidence: LoadedFormalEvidence, cohort: str,
                key: str, age_index: int) -> np.ndarray:
    return np.stack([
        np.asarray(evidence.raw_context[(cohort, seed)][key][age_index],
                   dtype=np.float64)
        for seed in evidence.contract.seeds
    ])


def _contrast(
        evidence: LoadedFormalEvidence, cohort: str, left_arm: str,
        right_arm: str, key: str, age_index: int, *, seed: int,
        confidence: float = 0.95, statistic: str = "difference"
        ) -> dict[str, Any]:
    reference = evidence.cells[(cohort, left_arm,
                                evidence.contract.seeds[0])]
    return paired_seed_cluster_bootstrap(
        _metric_matrix(evidence, cohort, left_arm, key, age_index),
        _metric_matrix(evidence, cohort, right_arm, key, age_index),
        reference["formal_test_label"][age_index],
        reference["formal_test_native_cluster_id"][age_index],
        draws=evidence.contract.bootstrap_draws, seed=seed,
        confidence=confidence, statistic=statistic)


def _resource_audit(
        evidence: LoadedFormalEvidence, cohort: str) -> dict[str, Any]:
    contract = evidence.contract
    target = contract.target_parameters[cohort]
    result: dict[str, Any] = {}
    baseline_flops = []
    for arm in contract.arms:
        reports = [evidence.resources[(cohort, arm, seed)]
                   for seed in contract.seeds]
        parameters = {int(report["trainable_parameters"])
                      for report in reports}
        flops = {float(report["forward_flops_per_episode"])
                 for report in reports}
        persistent = {int(report["persistent_state_floats"])
                      for report in reports}
        _require(len(parameters) == len(flops) == len(persistent) == 1,
                 f"resource contract varies across seeds: {cohort}/{arm}")
        count = parameters.pop()
        forward = flops.pop()
        state = persistent.pop()
        if arm == "none":
            _require(count == 0 and state == 0,
                     f"no-state arm has trainable/persistent state: {cohort}")
            result[arm] = {
                "trainable_parameters": count,
                "forward_flops_per_episode": forward,
                "persistent_state_floats": state,
                "parameter_matched": None,
                "flop_matched": None,
            }
            continue
        parameter_gap = abs(count - target) / target
        _require(parameter_gap <= contract.parameter_margin,
                 f"parameter fairness failed: {cohort}/{arm}")
        if arm in BASELINE_ARMS:
            baseline_flops.append(forward)
        result[arm] = {
            "trainable_parameters": count,
            "target_parameters": target,
            "parameter_relative_gap": parameter_gap,
            "parameter_matched": True,
            "forward_flops_per_episode": forward,
            "persistent_state_floats": state,
            "peak_cuda_bytes_mean": float(np.mean([
                report["peak_cuda_bytes"] for report in reports])),
            "wall_clock_train_seconds_mean": float(np.mean([
                report["wall_clock_train_seconds"] for report in reports])),
        }
    _require(baseline_flops and float(np.median(baseline_flops)) > 0,
             f"baseline FLOP reference is unavailable: {cohort}")
    median = float(np.median(baseline_flops))
    for arm in contract.arms:
        if arm == "none":
            continue
        gap = abs(result[arm]["forward_flops_per_episode"] - median) / median
        result[arm]["baseline_median_flops"] = median
        result[arm]["flop_relative_gap"] = gap
        result[arm]["flop_matched"] = gap <= contract.flop_margin
        _require(result[arm]["flop_matched"],
                 f"FLOP fairness failed: {cohort}/{arm}")
    return result


def analyze_formal_evidence(evidence: LoadedFormalEvidence) -> dict[str, Any]:
    """Run all registered per-age gates on already authenticated evidence."""

    contract = evidence.contract
    cohorts: dict[str, Any] = {}
    execution_passing_by_age = {str(age): 0 for age in contract.ages}
    for cohort_index, cohort in enumerate(contract.cohorts):
        comparators = evidence.comparators[cohort]
        _require(set(comparators) == {"retention", "next_feature", "execution"}
                 and all(arm in BASELINE_ARMS
                         for arm in comparators.values()),
                 f"locked comparator map is invalid: {cohort}")
        resource = _resource_audit(evidence, cohort)
        canonical = evidence.cells[(cohort, "sage_mem_full",
                                    contract.seeds[0])]
        labels = canonical["formal_test_label"][0]
        clusters = canonical["formal_test_native_cluster_id"][0]
        ages: dict[str, Any] = {}
        execution_presence = {
            EXECUTION_ARRAYS.issubset(
                evidence.cells[(cohort, arm, seed)])
            for arm in contract.arms for seed in contract.seeds
        }
        _require(len(execution_presence) == 1,
                 f"execution arrays are incomplete across cells: {cohort}")
        execution_supplied = execution_presence.pop()
        status = evidence.execution_status.get(cohort)
        if execution_supplied:
            random_vector = np.asarray(
                evidence.execution_random.get(cohort), dtype=np.float64)
            _require(isinstance(status, Mapping)
                     and status.get("eligible") is True
                     and isinstance(status.get("oracle_success"), (int, float))
                     and not isinstance(status.get("oracle_success"), bool)
                     and float(status["oracle_success"]) >=
                     contract.execution_oracle_gate
                     and isinstance(status.get("random_success"), (int, float))
                     and not isinstance(status.get("random_success"), bool)
                     and 0.0 <= float(status["random_success"]) <= 1.0
                     and random_vector.shape ==
                     (contract.formal_rows[cohort],)
                     and np.isin(random_vector, (0, 1)).all()
                     and math.isclose(float(random_vector.mean()),
                                      float(status["random_success"]),
                                      rel_tol=0.0, abs_tol=1e-12),
                     f"execution arrays bypass the oracle gate: {cohort}")
        elif status is not None:
            _require(isinstance(status, Mapping)
                     and status.get("eligible") is False,
                     f"eligible execution cohort omits arrays: {cohort}")
        if contract.require_raw_context:
            _require(all((cohort, seed) in evidence.raw_context
                         for seed in contract.seeds),
                     f"short-3/long-16 references are incomplete: {cohort}")
        for age_index, age in enumerate(contract.ages):
            namespace = (contract.bootstrap_seed + cohort_index * 10_000
                         + age_index * 1_000)
            zeros = np.zeros_like(_metric_matrix(
                evidence, cohort, "sage_mem_full",
                "formal_test_full_correct", age_index))
            host_accuracy = paired_seed_cluster_bootstrap(
                _metric_matrix(evidence, cohort, "sage_mem_full",
                               "formal_test_full_correct", age_index),
                zeros, labels, clusters, draws=contract.bootstrap_draws,
                seed=namespace)
            prior_accuracy = paired_seed_cluster_bootstrap(
                _metric_matrix(evidence, cohort, "sage_mem_full",
                               "formal_test_prior_correct", age_index),
                zeros, labels, clusters, draws=contract.bootstrap_draws,
                seed=namespace + 1)
            host_vs_comparator = _contrast(
                evidence, cohort, "sage_mem_full", comparators["retention"],
                "formal_test_full_correct", age_index,
                seed=namespace + 2)
            full_vs_reset = paired_seed_cluster_bootstrap(
                _metric_matrix(evidence, cohort, "sage_mem_full",
                               "formal_test_full_correct", age_index),
                _metric_matrix(evidence, cohort, "sage_mem_full",
                               "formal_test_reset_correct", age_index),
                labels, clusters, draws=contract.bootstrap_draws,
                seed=namespace + 3)
            full_vs_none = _contrast(
                evidence, cohort, "sage_mem_full", "none",
                "formal_test_full_correct", age_index,
                seed=namespace + 4)
            prior_vs_comparator = _contrast(
                evidence, cohort, "sage_mem_full", comparators["retention"],
                "formal_test_prior_correct", age_index,
                seed=namespace + 5)
            controls = {
                arm: _contrast(
                    evidence, cohort, "sage_mem_full", arm,
                    "formal_test_full_correct", age_index,
                    seed=namespace + 10 + index)
                for index, arm in enumerate(MECHANISM_CONTROLS)
            }
            mse = _contrast(
                evidence, cohort, "sage_mem_full",
                comparators["next_feature"], "formal_test_full_mse",
                age_index, seed=namespace + 20, confidence=0.90,
                statistic="relative_ratio")
            full_mse = _metric_matrix(
                evidence, cohort, "sage_mem_full",
                "formal_test_full_mse", age_index)
            reset_mse = _metric_matrix(
                evidence, cohort, "sage_mem_full",
                "formal_test_reset_mse", age_index)
            full_mse_mean = float(full_mse.mean())
            reset_ratio = (1.0 if full_mse_mean == 0.0
                           and float(reset_mse.mean()) == 0.0
                           else (float("inf") if full_mse_mean == 0.0
                                 else float(reset_mse.mean()) / full_mse_mean))
            gates = {
                "host_vs_locked_comparator":
                    host_vs_comparator["lower"] >=
                    contract.host_comparator_gain,
                "full_vs_reset": (
                    full_vs_reset["lower"] >= contract.reset_gain
                    and reset_ratio <= contract.reset_mse_ratio_max),
                "full_vs_none":
                    full_vs_none["lower"] >= contract.mechanism_gain,
                "all_mechanism_controls": all(
                    estimate["lower"] >= contract.mechanism_gain
                    for estimate in controls.values()),
                "next_mse_noninferiority":
                    mse["upper"] <= contract.mse_relative_margin,
            }
            raw_result = None
            if evidence.raw_context:
                short = _raw_matrix(
                    evidence, cohort, "formal_test_short_correct", age_index)
                long = _raw_matrix(
                    evidence, cohort, "formal_test_long_correct", age_index)
                raw_contrast = paired_seed_cluster_bootstrap(
                    long, short, labels, clusters,
                    draws=contract.bootstrap_draws, seed=namespace + 30)
                raw_result = {
                    "short3_accuracy": float(short.mean()),
                    "long16_accuracy": float(long.mean()),
                    "long16_minus_short3": raw_contrast,
                    "resolved_long_context_gain": raw_contrast["lower"] > 0,
                    "separate_from_parameter_matched_grid": True,
                }
            execution = None
            if execution_supplied:
                execution_comparator = comparators["execution"]
                full_exec = _metric_matrix(
                    evidence, cohort, "sage_mem_full",
                    "formal_test_full_execution_success", age_index)
                execution_status = evidence.execution_status[cohort]
                random_vector = np.asarray(
                    evidence.execution_random[cohort], dtype=np.float64)
                random_reference = np.repeat(
                    random_vector[None, :], len(contract.seeds), axis=0)
                execution = {
                    "full_vs_locked_comparator": _contrast(
                        evidence, cohort, "sage_mem_full",
                        execution_comparator,
                        "formal_test_full_execution_success", age_index,
                        seed=namespace + 40),
                    "full_vs_none": _contrast(
                        evidence, cohort, "sage_mem_full", "none",
                        "formal_test_full_execution_success", age_index,
                        seed=namespace + 41),
                    "full_vs_random": paired_seed_cluster_bootstrap(
                        full_exec, random_reference, labels, clusters,
                        draws=contract.bootstrap_draws, seed=namespace + 42),
                    "random_reference":
                        "sealed per-episode arm-blind random-success deck",
                    "random_reference_is_cohort_rate": False,
                    "oracle_success": execution_status.get("oracle_success"),
                    "random_success_mean": float(random_vector.mean()),
                }
                execution["pass"] = all(
                    execution[name]["lower"] >= contract.execution_gain
                    for name in ("full_vs_locked_comparator", "full_vs_none",
                                 "full_vs_random"))
                if execution["pass"]:
                    execution_passing_by_age[str(age)] += 1
            ages[str(age)] = {
                "primary_endpoint": "frozen-host full correctness",
                "host_full_accuracy": host_accuracy,
                "host_full_vs_locked_comparator": host_vs_comparator,
                "host_full_vs_reset": full_vs_reset,
                "host_full_vs_none": full_vs_none,
                "reset_to_full_mse_ratio": reset_ratio,
                "mechanism_controls": controls,
                "next_feature_relative_excess": mse,
                "gates": gates,
                "primary_host_claim_pass": all(gates.values()),
                "prior_diagnostic": {
                    "role": "diagnostic-only; cannot establish host use",
                    "accuracy": prior_accuracy,
                    "vs_locked_comparator": prior_vs_comparator,
                    "resolved_positive": prior_vs_comparator["lower"] > 0,
                    "enters_primary_host_claim": False,
                },
                "raw_context_reference": raw_result,
                "execution": execution,
            }
        cohorts[cohort] = {
            "locked_comparators": dict(comparators),
            "comparator_receipt": dict(
                evidence.comparator_receipts[cohort]),
            "backend_admission": dict(
                evidence.backend_admissions[cohort]),
            "resource_enforcement": resource,
            "ages": ages,
            "all_registered_ages_primary_host_claim_pass": all(
                value["primary_host_claim_pass"] for value in ages.values()),
            "execution_supplied": execution_supplied,
            "execution_pass_by_age": {
                age: (value["execution"]["pass"]
                      if value["execution"] is not None else None)
                for age, value in ages.items()},
        }
    eligible_execution = sum(
        1 for value in evidence.execution_status.values()
        if value.get("eligible") is True)
    return {
        "schema": AUDIT_SCHEMA,
        "study": "sage-mem-v1",
        "stage": "formal-evidence-audit",
        "status": "complete",
        "phase_a_cells_verified": evidence.phase_a_cells_verified,
        "finalized_cells_verified": evidence.finalized_cells_verified,
        "phase_a_grid_sha256": evidence.phase_a_grid_sha256,
        "identity_ledger_sha256": evidence.identity_ledger_sha256,
        "comparators_verified": len(evidence.comparators),
        "resources_verified": len(evidence.resources),
        "raw_context_references_verified":
            evidence.raw_context_references_verified,
        "bootstrap_draws_per_contrast": contract.bootstrap_draws,
        "cohorts": cohorts,
        "execution_program": {
            "optional": True,
            "eligible_cohorts": eligible_execution,
            "minimum_eligible_cohorts": 2,
            "program_claim_permitted": eligible_execution >= 2,
            "per_age": {
                age: {
                    "eligible_cohorts": eligible_execution,
                    "cohorts_passing": execution_passing_by_age[age],
                    "claim_permitted": eligible_execution >= 2,
                    "claim_pass": (eligible_execution >= 2 and
                                   execution_passing_by_age[age] >= 2),
                }
                for age in map(str, contract.ages)
            },
            "cross_age_conjunction_computed": False,
            "program_claim_pass": None,
        },
        "prior_can_substitute_for_host_output": False,
        "per_age_claims_only": True,
        "pooled_cross_host_score_computed": False,
        "universal_success_claim_permitted": False,
    }


def audit_formal_evidence(
        *, spec: Mapping[str, Any], phase_a_root: str | Path,
        finalized_root: str | Path, prepare_root: str | Path,
        raw_context_root: str | Path | None = None) -> dict[str, Any]:
    contract = contract_from_spec(spec, require_raw_context=True)
    evidence = load_formal_evidence(
        spec=spec, phase_a_root=phase_a_root,
        finalized_root=finalized_root, prepare_root=prepare_root,
        raw_context_root=raw_context_root, contract=contract)
    return analyze_formal_evidence(evidence)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--phase-a-root", type=Path, required=True)
    parser.add_argument("--finalized-root", type=Path, required=True)
    parser.add_argument("--prepare-root", type=Path, required=True)
    parser.add_argument(
        "--raw-context-root", type=Path,
        help=("label-free raw-context Phase-A root; defaults to "
              "<prepare-root>/../raw_context_phase_a"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--resume", action="store_true",
        help=("recompute the complete audit and accept an existing --output "
              "only when its canonical report is identical"))
    return parser.parse_args(argv)


def _publish_or_validate_report(
        path: Path, report: Mapping[str, Any], *, resume: bool) -> str:
    """Create one report, or fail closed while authenticating a resume."""

    if path.exists() or path.is_symlink():
        _require(resume,
                 f"refusing to overwrite existing audit report: {path}")
        _require(path.is_file() and not path.is_symlink(),
                 f"resume audit report is absent or unsafe: {path}")
        existing = _read_json(path, "existing formal evidence audit report")
        _require(_canonical_json(existing) == _canonical_json(report),
                 "resume audit report differs from freshly authenticated "
                 "evidence")
        return "validated-existing"
    _atomic_json(path, report)
    return "created"


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    _require(not args.resume or (args.execute and args.output is not None),
             "--resume requires --execute and --output")
    from scripts.sage_mem_v1_spec import load_spec
    spec = load_spec(args.spec, verify_parent_paths=False)
    if not args.execute:
        print(_canonical_json({
            "schema": AUDIT_SCHEMA,
            "preview": True,
            "phase_a_cells_required": 600,
            "finalized_cells_required": 600,
            "ages": list(AGES),
            "primary_endpoint": "frozen-host full correctness",
            "prior_role": "diagnostic-only",
            "pooled_cross_host_score": False,
            "no_outcomes_read": True,
            "no_jobs_launched": True,
        }))
        return 0
    report = audit_formal_evidence(
        spec=spec, phase_a_root=args.phase_a_root,
        finalized_root=args.finalized_root,
        prepare_root=args.prepare_root,
        raw_context_root=args.raw_context_root)
    if args.output is not None:
        _publish_or_validate_report(
            args.output, report, resume=args.resume)
    print(_canonical_json(report))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FormalEvidenceAuditError as error:
        print(f"formal evidence audit stopped: {error}", file=sys.stderr)
        raise SystemExit(2) from error


__all__ = [
    "AUDIT_SCHEMA", "COHORTS", "ARMS", "SEEDS", "AGES",
    "BASELINE_ARMS", "MECHANISM_CONTROLS", "FormalEvidenceAuditError",
    "AuditContract", "LoadedFormalEvidence", "contract_from_spec",
    "paired_seed_cluster_bootstrap", "load_formal_evidence",
    "analyze_formal_evidence", "audit_formal_evidence", "parse_args", "main",
]
