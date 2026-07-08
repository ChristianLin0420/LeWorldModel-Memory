#!/usr/bin/env python3
"""Fail-closed stage and cell runner for the preregistered SAGE-Mem v1 audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sage_mem_v1_interface import (  # noqa: E402
    SageMemInterfaceError,
    integration_requirements,
    load_host_adapter_contract,
    load_model_contract,
    validate_host_adapter_instance,
)
from scripts.sage_mem_v1_spec import (  # noqa: E402
    ARMS,
    COHORTS,
    DEFAULT_SPEC,
    DEVELOPMENT_ARMS,
    DEVELOPMENT_SEEDS,
    FORMAL_SEEDS,
    cell_directory,
    canonical_json,
    development_cell_directory,
    load_spec,
    output_root,
    resolve_repo_path,
    sha256_file,
    spec_fingerprint,
)


PROTOCOL_PRODUCERS = (
    "configs/sage_mem_v1.yaml",
    "scripts/sage_mem_v1_spec.py",
    "scripts/sage_mem_v1_interface.py",
    "scripts/sage_mem_v1_losses.py",
    "scripts/prepare_sage_mem_v1_development.py",
    "scripts/run_sage_mem_v1.py",
    "scripts/launch_sage_mem_v1.py",
    "scripts/audit_sage_mem_v1.py",
    "tests/test_sage_mem_v1_protocol.py",
)
FORMAL_CONFIRMATION = "RUN_SAGE_MEM_V1_FORMAL"


class SageMemRunError(RuntimeError):
    """A stage cannot proceed without weakening the sealed protocol."""


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value)


def _clear_dead_partial_directories(final: Path) -> None:
    """Remove only dead, runner-owned atomic staging dirs during --resume."""
    prefix = f".{final.name}.partial-"
    if not final.parent.exists():
        return
    for path in final.parent.iterdir():
        if not path.is_dir() or not path.name.startswith(prefix):
            continue
        remainder = path.name[len(prefix):]
        pid_text = remainder.split("-", 1)[0]
        if not pid_text.isdigit():
            raise SageMemRunError(f"malformed partial directory: {path}")
        pid = int(pid_text)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            shutil.rmtree(path)
        except PermissionError as error:
            raise SageMemRunError(
                f"cannot determine partial-directory owner: {path}") from error
        else:
            raise SageMemRunError(
                f"development/formal cell still has a live writer: {path}")


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(value) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _receipt_path(spec: Mapping[str, Any], stage: str,
                  cohort: str | None = None) -> Path:
    suffix = f"/{cohort}" if cohort is not None else ""
    return output_root(spec) / "receipts" / f"{stage}{suffix}/receipt.json"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise SageMemRunError(f"JSON root must be a mapping: {path}")
    return value


def validate_manifest_identity(value: Mapping[str, Any], spec: Mapping[str, Any],
                               *, stage: str) -> None:
    if value.get("schema_version") != 1 \
            or value.get("study") != "sage-mem-v1" \
            or value.get("stage") != stage \
            or value.get("protocol_fingerprint") != spec_fingerprint(spec):
        raise SageMemRunError(f"invalid or stale {stage} manifest")


def preflight_report(spec: Mapping[str, Any], *,
                     require_integrations: bool) -> dict[str, Any]:
    parent_identities: dict[str, dict[str, Any]] = {}
    for cohort, record in spec["cohorts"].items():
        protocol = resolve_repo_path(record["parent_protocol"])
        if not protocol.is_file():
            raise SageMemRunError(f"parent protocol missing: {protocol}")
        exclusions = []
        for relative in record["forbidden_parent_artifacts"]:
            path = resolve_repo_path(relative)
            if not path.is_file():
                raise SageMemRunError(f"parent exclusion artifact missing: {path}")
            exclusions.append({
                "path": relative, "size": path.stat().st_size,
                "sha256": sha256_file(path),
            })
        parent_identities[cohort] = {
            "protocol": str(protocol.relative_to(ROOT)),
            "protocol_sha256": sha256_file(protocol),
            "exclusion_artifacts": exclusions,
        }
    integrations: dict[str, Any] = {
        "model": "unavailable", "host_adapter": "unavailable"}
    integration_errors: list[str] = []
    try:
        model = load_model_contract(spec["model_interface"])
        path = Path(model.module.__file__).resolve()
        integrations["model"] = {
            "path": str(path), "size": path.stat().st_size,
            "sha256": sha256_file(path)}
    except SageMemInterfaceError as error:
        integration_errors.append(str(error))
    try:
        host = load_host_adapter_contract(spec["host_adapter_interface"])
        path = Path(host.module.__file__).resolve()
        integrations["host_adapter"] = {
            "path": str(path), "size": path.stat().st_size,
            "sha256": sha256_file(path)}
    except SageMemInterfaceError as error:
        integration_errors.append(str(error))
    integration_error = "; ".join(integration_errors) or None
    if integration_error is not None and require_integrations:
        raise SageMemRunError(integration_error)
    return {
        "schema_version": 1,
        "study": "sage-mem-v1",
        "stage": "preflight",
        "status": "ready" if integration_error is None else "blocked-integration",
        "protocol_fingerprint": spec_fingerprint(spec),
        "spec_sha256": spec["_spec_sha256"],
        "parent_identities": parent_identities,
        "seed_registry": spec["_seed_registry"],
        "integrations": integrations,
        "integration_error": integration_error,
        "integration_requirements": integration_requirements(),
        "formal_execution_started": False,
    }


def require_exact_gpu(spec: Mapping[str, Any], cohort: str) -> None:
    physical = int(spec["cohorts"][cohort]["gpu"])
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(physical):
        raise SageMemRunError(
            f"{cohort} owns physical GPU {physical}; CUDA_VISIBLE_DEVICES "
            f"must be exactly {physical!r}, got {visible!r}")
    try:
        import torch
    except ImportError as error:
        raise SageMemRunError("torch is unavailable") from error
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SageMemRunError("worker must see exactly one available CUDA device")


def _integration_contracts(spec: Mapping[str, Any]) -> tuple[Any, Any]:
    try:
        return (load_model_contract(spec["model_interface"]),
                load_host_adapter_contract(spec["host_adapter_interface"]))
    except SageMemInterfaceError as error:
        raise SageMemRunError(str(error)) from error


def _adapter(spec: Mapping[str, Any], cohort: str) -> tuple[Any, Any]:
    model_contract, host_contract = _integration_contracts(spec)
    adapter = host_contract.builder(cohort=cohort, spec=spec)
    try:
        description = validate_host_adapter_instance(
            adapter, cohort=cohort,
            api_version=spec["host_adapter_interface"]["api_version"])
    except SageMemInterfaceError as error:
        raise SageMemRunError(str(error)) from error
    expected_parameters = spec["cohorts"][cohort]["target_parameters"]
    from lewm.models.sage_mem import sage_mem_parameter_count
    achieved = sage_mem_parameter_count(
        description["embed_dim"], description["action_dim"])
    if achieved != expected_parameters:
        raise SageMemRunError(
            f"host dimensions imply {achieved} SAGE parameters, expected "
            f"{expected_parameters} for {cohort}")
    return model_contract, adapter


def _smoke(spec: Mapping[str, Any], cohort: str) -> dict[str, Any]:
    _require_ready_receipt(spec, "preflight")
    require_exact_gpu(spec, cohort)
    model_contract, adapter = _adapter(spec, cohort)
    result = adapter.smoke(model_contract=model_contract)
    if not isinstance(result, Mapping) or result.get("status") != "passed" \
            or result.get("labels_used") is not False \
            or result.get("gradient_finite") is not True \
            or result.get("reset_isolates_state") is not True:
        raise SageMemRunError(f"smoke failed for {cohort}")
    return {
        "schema_version": 1, "study": "sage-mem-v1", "stage": "smoke",
        "status": "passed", "cohort": cohort,
        "physical_gpu": spec["cohorts"][cohort]["gpu"],
        "protocol_fingerprint": spec_fingerprint(spec),
        "result": dict(result), "formal_execution_started": False,
    }


def _producer_identities() -> dict[str, dict[str, Any]]:
    identities = {}
    for relative in PROTOCOL_PRODUCERS:
        path = resolve_repo_path(relative)
        if not path.is_file():
            raise SageMemRunError(f"protocol producer missing: {relative}")
        identities[relative] = {
            "size": path.stat().st_size, "sha256": sha256_file(path)}
    return identities


def _seal(spec: Mapping[str, Any]) -> dict[str, Any]:
    _require_ready_receipt(spec, "preflight")
    for cohort in COHORTS:
        path = _receipt_path(spec, "smoke", cohort)
        if not path.is_file():
            raise SageMemRunError(f"smoke receipt missing: {cohort}")
        smoke = _load_json(path)
        validate_manifest_identity(smoke, spec, stage="smoke")
        if smoke.get("status") != "passed":
            raise SageMemRunError(f"smoke did not pass: {cohort}")
    development_audit_path = (
        output_root(spec) / "development" / "audit_receipt.json")
    if not development_audit_path.is_file():
        raise SageMemRunError("development audit receipt missing")
    development_audit = _load_json(development_audit_path)
    validate_manifest_identity(
        development_audit, spec, stage="development-audit")
    if development_audit.get("status") != "complete" \
            or development_audit.get("registered_cells_verified") != \
            len(COHORTS) * len(DEVELOPMENT_ARMS) * len(DEVELOPMENT_SEEDS):
        raise SageMemRunError("development audit is incomplete")
    selections = development_audit.get("selection_receipts")
    if not isinstance(selections, Mapping) or set(selections) != set(COHORTS):
        raise SageMemRunError("development selection identities missing")
    for cohort, identity in selections.items():
        if not isinstance(identity, Mapping) \
                or not _is_sha256(identity.get("sha256")):
            raise SageMemRunError(
                f"development selection identity malformed: {cohort}")
        selection_path = resolve_repo_path(identity.get("path"))
        if not selection_path.is_file() \
                or selection_path.stat().st_size != identity.get("size") \
                or sha256_file(selection_path) != identity["sha256"]:
            raise SageMemRunError(
                f"development selection receipt changed: {cohort}")
    model_contract, host_contract = _integration_contracts(spec)
    integrations = {}
    for name, module in (("model", model_contract.module),
                         ("host_adapter", host_contract.module)):
        path = Path(module.__file__).resolve()
        integrations[name] = {
            "path": str(path), "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return {
        "schema_version": 1, "study": "sage-mem-v1", "stage": "seal",
        "status": "sealed", "protocol_fingerprint": spec_fingerprint(spec),
        "spec_sha256": spec["_spec_sha256"],
        "producer_identities": _producer_identities(),
        "integration_identities": integrations,
        "development_audit": {
            "path": str(development_audit_path.relative_to(ROOT)),
            "size": development_audit_path.stat().st_size,
            "sha256": sha256_file(development_audit_path),
        },
        "seed_registry": spec["_seed_registry"],
        "formal_execution_started": False,
    }


def require_valid_lock(spec: Mapping[str, Any]) -> dict[str, Any]:
    path = resolve_repo_path(spec["implementation_lock"])
    if not path.is_file():
        raise SageMemRunError("sealed implementation lock is missing")
    lock = _load_json(path)
    validate_manifest_identity(lock, spec, stage="seal")
    if lock.get("status") != "sealed":
        raise SageMemRunError("implementation lock is not sealed")
    for relative, identity in lock["producer_identities"].items():
        path = resolve_repo_path(relative)
        if path.stat().st_size != identity["size"] \
                or sha256_file(path) != identity["sha256"]:
            raise SageMemRunError(f"sealed producer changed: {relative}")
    for identity in lock["integration_identities"].values():
        path = Path(identity["path"])
        if not path.is_file() or path.stat().st_size != identity["size"] \
                or sha256_file(path) != identity["sha256"]:
            raise SageMemRunError(f"sealed integration changed: {path}")
    audit_identity = lock.get("development_audit")
    if not isinstance(audit_identity, Mapping) \
            or not _is_sha256(audit_identity.get("sha256")):
        raise SageMemRunError("sealed development audit identity is missing")
    audit_path = resolve_repo_path(audit_identity.get("path"))
    if not audit_path.is_file() or audit_path.stat().st_size != \
            audit_identity.get("size") or sha256_file(audit_path) != \
            audit_identity["sha256"]:
        raise SageMemRunError("sealed development audit changed")
    audit = _load_json(audit_path)
    selection_receipts = audit.get("selection_receipts")
    if not isinstance(selection_receipts, Mapping) \
            or set(selection_receipts) != set(COHORTS):
        raise SageMemRunError("sealed development selections are incomplete")
    for cohort, identity in selection_receipts.items():
        if not isinstance(identity, Mapping) \
                or not _is_sha256(identity.get("sha256")):
            raise SageMemRunError(
                f"sealed development selection malformed: {cohort}")
        selection_path = resolve_repo_path(identity.get("path"))
        if not selection_path.is_file() \
                or selection_path.stat().st_size != identity.get("size") \
                or sha256_file(selection_path) != identity.get("sha256"):
            raise SageMemRunError(
                f"sealed development selection changed: {cohort}")
    return lock


def _development_bank_path(spec: Mapping[str, Any], cohort: str) -> Path:
    return (output_root(spec) / "development_banks" / cohort
            / "manifest.json")


def validate_development_bank(spec: Mapping[str, Any], cohort: str,
                              value: Mapping[str, Any]) -> None:
    selection = value.get("selection")
    expected_count = spec["cohorts"][cohort]["split_episodes"]["development"]
    seed_key = f"{cohort}/development/episode_selection"
    if value.get("schema_version") != 1 \
            or value.get("study") != "sage-mem-v1" \
            or value.get("stage") != "development-bank" \
            or value.get("status") != "prepared-parent-train-only" \
            or value.get("cohort") != cohort \
            or value.get("protocol_fingerprint") != spec_fingerprint(spec) \
            or value.get("parent_train_only") is not True \
            or value.get("parent_validation_or_test_read") is not False \
            or value.get("semantic_labels_read_for_selection") is not False \
            or value.get("formal_evidence_permitted") is not False \
            or not isinstance(selection, Mapping) \
            or selection.get("count") != expected_count \
            or selection.get("seed") != spec["_seed_registry"][seed_key] \
            or not _is_sha256(selection.get("sha256")):
        raise SageMemRunError(f"invalid development bank: {cohort}")
    rows = selection.get("rows")
    episodes = selection.get("episode_indices")
    starts = selection.get("local_starts")
    if any(not isinstance(item, list) or len(item) != expected_count
           for item in (rows, episodes, starts)) \
            or len(set(rows)) != expected_count:
        raise SageMemRunError(f"development bank selection malformed: {cohort}")
    source = value.get("source")
    if not isinstance(source, Mapping):
        raise SageMemRunError(f"development source identity missing: {cohort}")
    source_path = resolve_repo_path(source.get("path"))
    if not source_path.is_file() or source_path.stat().st_size != \
            source.get("size") or sha256_file(source_path) != source.get("sha256"):
        raise SageMemRunError(f"development source changed: {cohort}")


def validate_development_payload(
        spec: Mapping[str, Any], cohort: str, arm: str, seed: int,
        value: Mapping[str, Any], *, bank_sha256: str) -> None:
    if value.get("status") != "complete" \
            or value.get("cohort") != cohort or value.get("arm") != arm \
            or value.get("seed") != seed \
            or value.get("labels_used_for_training") is not False \
            or value.get("formal_data_read") is not False \
            or not _is_sha256(value.get("host_hash_before")) \
            or value.get("host_hash_before") != value.get("host_hash_after") \
            or value.get("development_bank_sha256") != bank_sha256 \
            or not isinstance(value.get("gradient_finite"), bool):
        raise SageMemRunError(
            "development cell identity/label/data/host invariant failed")
    required = spec["fairness_reporting"]["required_per_arm_fields"]
    resources = value.get("resource_report")
    if not isinstance(resources, Mapping) or set(required).difference(resources):
        raise SageMemRunError("development cell lacks resource reporting")
    for key in required:
        number = resources[key]
        if isinstance(number, bool) or not isinstance(number, (int, float)) \
                or not math.isfinite(float(number)) or number < 0:
            raise SageMemRunError(f"invalid development resource field: {key}")
    metrics = value.get("development_metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != {
            "next_feature_mse", "retention_balanced_accuracy",
            "execution_success"}:
        raise SageMemRunError("development selection metrics are incomplete")
    for name, number in metrics.items():
        if isinstance(number, bool) or not isinstance(number, (int, float)) \
                or not math.isfinite(float(number)):
            raise SageMemRunError(f"development metric is not finite: {name}")
    if metrics["next_feature_mse"] < 0 \
            or not 0 <= metrics["retention_balanced_accuracy"] <= 1 \
            or not 0 <= metrics["execution_success"] <= 1:
        raise SageMemRunError("development metrics leave registered ranges")
    artifacts = value.get("artifacts", [])
    if not isinstance(artifacts, list):
        raise SageMemRunError("development artifacts must be a list")
    paths = []
    for record in artifacts:
        if not isinstance(record, Mapping) \
                or not isinstance(record.get("path"), str) \
                or not _is_sha256(record.get("sha256")):
            raise SageMemRunError("development artifact identity malformed")
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts \
                or not relative.parts:
            raise SageMemRunError("development artifact path is unsafe")
        paths.append(relative)
    if len(set(paths)) != len(paths):
        raise SageMemRunError("duplicate development artifact path")


def _normalize_development_result(
        value: Mapping[str, Any], *, spec: Mapping[str, Any], cohort: str,
        arm: str, seed: int, bank: Mapping[str, Any], bank_sha256: str
        ) -> dict[str, Any]:
    """Validate the adapter-native receipt, then expose audit-stable fields."""
    if value.get("status") == "complete" \
            and isinstance(value.get("development_metrics"), Mapping):
        return dict(value)
    manifest_digest = hashlib.sha256(
        canonical_json(bank).encode("utf-8")).hexdigest()
    if value.get("status") not in {"complete", "complete-development"} \
            or value.get("cohort") != cohort or value.get("arm") != arm \
            or value.get("seed") != seed \
            or value.get("development_only") is not True \
            or value.get("formal_evidence_permitted") is not False \
            or value.get("parent_train_only") is not True \
            or value.get("labels_used_for_training") is not False \
            or value.get("labels_used_for_posthoc_readout") is not True \
            or value.get("development_manifest_sha256") != manifest_digest \
            or not isinstance(value.get("gradient_finite"), bool) \
            or not _is_sha256(value.get("host_hash_before")) \
            or value.get("host_hash_before") != value.get("host_hash_after"):
        raise SageMemRunError(
            "adapter-native development receipt violates identity/data boundary")
    ages = value.get("ages")
    if not isinstance(ages, Mapping) or set(ages) != {
            str(age) for age in spec["cohorts"][cohort]["ages"]}:
        raise SageMemRunError("adapter development ages are incomplete")
    retention = []
    for age in spec["cohorts"][cohort]["ages"]:
        record = ages[str(age)]
        required = {
            "host_output_balanced_accuracy", "prior_balanced_accuracy",
            "reset_with_full_readout_balanced_accuracy",
            "full_next_feature_mse", "reset_next_feature_mse",
            "reset_to_full_mse_ratio", "readout_fit_parent_train_rows",
            "readout_eval_parent_train_rows",
        }
        if not isinstance(record, Mapping) or required.difference(record):
            raise SageMemRunError(f"adapter age metrics incomplete: {age}")
        for name in required.difference({
                "readout_fit_parent_train_rows",
                "readout_eval_parent_train_rows"}):
            number = record[name]
            if isinstance(number, bool) or not isinstance(number, (int, float)) \
                    or not math.isfinite(float(number)):
                raise SageMemRunError(
                    f"adapter age metric is not finite: {age}/{name}")
        for name in (
                "host_output_balanced_accuracy", "prior_balanced_accuracy",
                "reset_with_full_readout_balanced_accuracy"):
            if not 0 <= record[name] <= 1:
                raise SageMemRunError(
                    f"adapter accuracy leaves [0,1]: {age}/{name}")
        for name in ("full_next_feature_mse", "reset_next_feature_mse",
                     "reset_to_full_mse_ratio"):
            if record[name] < 0:
                raise SageMemRunError(
                    f"adapter MSE metric is negative: {age}/{name}")
        retention.append(float(record["prior_balanced_accuracy"]))
    next_mse = value.get("next_feature_mse")
    if isinstance(next_mse, bool) or not isinstance(next_mse, (int, float)) \
            or not math.isfinite(float(next_mse)) or next_mse < 0:
        raise SageMemRunError("adapter next-feature MSE is invalid")
    resources = value.get("resource_report")
    if not isinstance(resources, Mapping):
        raise SageMemRunError("adapter resource report is missing")
    artifacts = []
    for name in ("episode_results", "checkpoint"):
        record = value.get(name)
        if not isinstance(record, Mapping) \
                or not isinstance(record.get("path"), str) \
                or not _is_sha256(record.get("sha256")):
            raise SageMemRunError(f"adapter {name} identity is malformed")
        artifacts.append({"path": record["path"], "sha256": record["sha256"]})
    result = dict(value)
    result.update({
        "adapter_status": value["status"],
        "status": "complete",
        "formal_data_read": False,
        "development_bank_sha256": bank_sha256,
        "development_metrics": {
            "next_feature_mse": float(next_mse),
            "retention_balanced_accuracy": sum(retention) / len(retention),
            # No development consumer is run. The retention comparator is
            # explicitly reused for execution rather than selecting on a
            # fabricated development execution metric.
            "execution_success": sum(retention) / len(retention),
        },
        "execution_metric_status": (
            "not-measured; retention comparator reused by preregistration"),
        "artifacts": artifacts,
    })
    return result


def validate_development_cell_directory(
        spec: Mapping[str, Any], path: Path, cohort: str, arm: str,
        seed: int) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise SageMemRunError(f"development cell manifest missing: {path}")
    manifest = _load_json(manifest_path)
    validate_manifest_identity(manifest, spec, stage="development")
    bank_path = _development_bank_path(spec, cohort)
    if not bank_path.is_file():
        raise SageMemRunError(f"development bank missing: {cohort}")
    bank_sha = sha256_file(bank_path)
    validate_development_payload(
        spec, cohort, arm, seed, manifest["result"], bank_sha256=bank_sha)
    allowed = {Path("manifest.json")}
    for record in manifest["result"].get("artifacts", []):
        relative = Path(record["path"])
        artifact = path / relative
        if not artifact.is_file() or sha256_file(artifact) != record["sha256"]:
            raise SageMemRunError(
                f"development artifact identity failed: {artifact}")
        allowed.add(relative)
    actual = {item.relative_to(path) for item in path.rglob("*")
              if item.is_file()}
    if actual != allowed:
        raise SageMemRunError(
            "unexpected development cell files: "
            f"{sorted(map(str, actual.difference(allowed)))}")
    return manifest


def _require_ready_receipt(spec: Mapping[str, Any], stage: str,
                           cohort: str | None = None) -> dict[str, Any]:
    path = _receipt_path(spec, stage, cohort)
    if not path.is_file():
        suffix = f": {cohort}" if cohort else ""
        raise SageMemRunError(f"{stage} receipt missing{suffix}")
    value = _load_json(path)
    validate_manifest_identity(value, spec, stage=stage)
    expected_status = "ready" if stage == "preflight" else "passed"
    if value.get("status") != expected_status:
        raise SageMemRunError(f"{stage} receipt is not {expected_status}")
    if stage == "preflight":
        integrations = value.get("integrations")
        if not isinstance(integrations, Mapping):
            raise SageMemRunError("preflight integration identities missing")
        for name in ("model", "host_adapter"):
            identity = integrations.get(name)
            if not isinstance(identity, Mapping) \
                    or not _is_sha256(identity.get("sha256")):
                raise SageMemRunError(
                    f"preflight integration identity malformed: {name}")
            path = Path(identity.get("path", ""))
            if not path.is_file() or path.stat().st_size != identity.get("size") \
                    or sha256_file(path) != identity["sha256"]:
                raise SageMemRunError(
                    f"integration changed after preflight: {name}")
    return value


def _run_development_cell(
        spec: Mapping[str, Any], cohort: str, arm: str, seed: int, *,
        resume: bool) -> dict[str, Any]:
    _require_ready_receipt(spec, "preflight")
    _require_ready_receipt(spec, "smoke", cohort)
    require_exact_gpu(spec, cohort)
    bank_path = _development_bank_path(spec, cohort)
    if not bank_path.is_file():
        raise SageMemRunError(
            f"development bank missing; run deterministic builder: {cohort}")
    bank = _load_json(bank_path)
    validate_development_bank(spec, cohort, bank)
    bank_sha = sha256_file(bank_path)
    final = development_cell_directory(spec, cohort, arm, seed)
    if resume:
        _clear_dead_partial_directories(final)
    if final.exists():
        if not resume:
            raise FileExistsError(f"development cell already exists: {final}")
        return validate_development_cell_directory(
            spec, final, cohort, arm, seed)
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = final.parent / (
        f".{final.name}.partial-{os.getpid()}-{uuid.uuid4().hex}")
    staging.mkdir(mode=0o750)
    try:
        model_contract, adapter = _adapter(spec, cohort)
        raw_result = adapter.run_development_cell(
            arm=arm, seed=seed, output_directory=staging,
            model_contract=model_contract, development_manifest=bank,
        )
        if not isinstance(raw_result, Mapping):
            raise SageMemRunError(
                "run_development_cell must return a mapping")
        result = _normalize_development_result(
            raw_result, spec=spec, cohort=cohort, arm=arm, seed=seed,
            bank=bank, bank_sha256=bank_sha)
        validate_development_payload(
            spec, cohort, arm, seed, result, bank_sha256=bank_sha)
        for record in result.get("artifacts", []):
            artifact = staging / record["path"]
            if not artifact.is_file() or sha256_file(artifact) != \
                    record["sha256"]:
                raise SageMemRunError(
                    f"development artifact hash mismatch: {artifact}")
        manifest = {
            "schema_version": 1, "study": "sage-mem-v1",
            "stage": "development", "status": "complete",
            "cohort": cohort, "arm": arm, "seed": seed,
            "physical_gpu": spec["cohorts"][cohort]["gpu"],
            "protocol_fingerprint": spec_fingerprint(spec),
            "development_bank_sha256": bank_sha,
            "result": dict(result), "completed_unix_ns": time.time_ns(),
            "formal_execution_started": False,
        }
        atomic_json(staging / "manifest.json", manifest)
        os.rename(staging, final)
        return validate_development_cell_directory(
            spec, final, cohort, arm, seed)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_prepared_payload(spec: Mapping[str, Any], cohort: str,
                              value: Mapping[str, Any]) -> None:
    expected = spec["cohorts"][cohort]["split_episodes"]
    if value.get("status") != "prepared" \
            or value.get("cohort") != cohort \
            or value.get("disjoint_with_every_parent_bank") is not True \
            or value.get("development_formal_disjoint") is not True \
            or value.get("formal_labels_hidden") is not True \
            or value.get("labels_used_for_carrier_training") is not False \
            or not isinstance(value.get("gdelta_development_healthy"), bool):
        raise SageMemRunError(f"fresh-bank contract failed: {cohort}")
    comparators = value.get("locked_comparators")
    baseline_arms = {
        "gru", "lstm", "ssm", "fixed_trust", "gdelta",
        "fixed_trust_aux", "ssm_aux",
    }
    if not isinstance(comparators, Mapping) or set(comparators) != {
            "retention", "next_feature", "execution"} \
            or any(arm not in baseline_arms for arm in comparators.values()):
        raise SageMemRunError(
            f"development comparators are not valid and locked: {cohort}")
    selection_path = (output_root(spec) / "development" / "selections"
                      / cohort / "receipt.json")
    if not selection_path.is_file():
        raise SageMemRunError(
            f"development selection receipt missing: {cohort}")
    selection = _load_json(selection_path)
    validate_manifest_identity(
        selection, spec, stage="development-selection")
    if selection.get("status") != "selected" \
            or selection.get("locked_comparators") != dict(comparators) \
            or selection.get("gdelta_development_healthy") != value.get(
                "gdelta_development_healthy"):
        raise SageMemRunError(
            f"formal preparation changed development selection: {cohort}")
    if value.get("gdelta_development_healthy") is False \
            and "gdelta" in comparators.values():
        raise SageMemRunError(
            f"unhealthy gDelta selected as formal comparator: {cohort}")
    splits = value.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != set(expected):
        raise SageMemRunError(f"fresh-bank split registry malformed: {cohort}")
    hashes = set()
    for split, count in expected.items():
        record = splits[split]
        if not isinstance(record, Mapping) or record.get("count") != count \
                or not isinstance(record.get("selection_sha256"), str) \
                or len(record["selection_sha256"]) != 64:
            raise SageMemRunError(f"fresh-bank split invalid: {cohort}/{split}")
        hashes.add(record["selection_sha256"])
    if len(hashes) != len(expected):
        raise SageMemRunError(f"fresh-bank split hashes collide: {cohort}")


def _prepare(spec: Mapping[str, Any], cohort: str) -> dict[str, Any]:
    if spec["freshness"]["formal_preparation_status"] == \
            "pending-executable-fresh-bank-builders":
        raise SageMemRunError(
            "formal preparation is fail-closed: executable fresh-bank "
            "builders have not yet been delivered, reviewed, and registered")
    require_valid_lock(spec)
    require_exact_gpu(spec, cohort)
    model_contract, adapter = _adapter(spec, cohort)
    result = adapter.prepare_fresh_banks(
        split_counts=spec["cohorts"][cohort]["split_episodes"],
        seed_registry=spec["_seed_registry"],
        forbidden_parent_artifacts=spec["cohorts"][cohort][
            "forbidden_parent_artifacts"],
        model_contract=model_contract,
    )
    if not isinstance(result, Mapping):
        raise SageMemRunError("prepare_fresh_banks must return a mapping")
    validate_prepared_payload(spec, cohort, result)
    return {
        "schema_version": 1, "study": "sage-mem-v1", "stage": "prepare",
        "status": "prepared", "cohort": cohort,
        "protocol_fingerprint": spec_fingerprint(spec),
        "result": dict(result), "formal_execution_started": False,
    }


def validate_formal_payload(spec: Mapping[str, Any], cohort: str, arm: str,
                            seed: int, value: Mapping[str, Any]) -> None:
    if value.get("status") != "complete" \
            or value.get("cohort") != cohort or value.get("arm") != arm \
            or value.get("seed") != seed \
            or value.get("labels_used_for_training") is not False \
            or not _is_sha256(value.get("host_hash_before")) \
            or value.get("host_hash_before") != value.get("host_hash_after"):
        raise SageMemRunError("formal cell identity/label/host invariant failed")
    required = spec["fairness_reporting"]["required_per_arm_fields"]
    resources = value.get("resource_report")
    if not isinstance(resources, Mapping) or set(required).difference(resources):
        raise SageMemRunError("formal cell lacks resource reporting")
    for key in required:
        number = resources[key]
        if isinstance(number, bool) or not isinstance(number, (int, float)) \
                or not math.isfinite(float(number)) or number < 0:
            raise SageMemRunError(f"invalid resource field: {key}")
    if isinstance(value.get("next_feature_mse"), bool) \
            or not isinstance(value.get("next_feature_mse"), (int, float)) \
            or not math.isfinite(float(value["next_feature_mse"])) \
            or value["next_feature_mse"] < 0:
        raise SageMemRunError("next-feature health metric missing")
    required_flags = (
        "host_output_exposure_measured", "reset_intervention_measured",
        "external_consumer_gate_evaluated", "counterfactual_pairing_preserved",
    )
    if any(value.get(flag) is not True for flag in required_flags):
        raise SageMemRunError("formal causal/execution measurements incomplete")
    artifact = value.get("episode_results")
    if not isinstance(artifact, Mapping) \
            or not isinstance(artifact.get("path"), str) \
            or not _is_sha256(artifact.get("sha256")):
        raise SageMemRunError("episode-level result artifact missing")
    relative = Path(artifact["path"])
    if relative.is_absolute() or ".." in relative.parts \
            or len(relative.parts) != 1:
        raise SageMemRunError("formal episode artifact path is unsafe")


def validate_cell_directory(spec: Mapping[str, Any], path: Path,
                            cohort: str, arm: str, seed: int) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise SageMemRunError(f"cell manifest missing: {path}")
    manifest = _load_json(manifest_path)
    validate_manifest_identity(manifest, spec, stage="full")
    validate_formal_payload(spec, cohort, arm, seed, manifest["result"])
    artifact = path / manifest["result"]["episode_results"]["path"]
    if not artifact.is_file() \
            or sha256_file(artifact) != manifest["result"][
                "episode_results"]["sha256"]:
        raise SageMemRunError(f"cell artifact identity failed: {path}")
    allowed = {"manifest.json", artifact.name}
    unexpected = {item.name for item in path.iterdir()}.difference(allowed)
    if unexpected:
        raise SageMemRunError(f"unexpected formal cell files: {sorted(unexpected)}")
    return manifest


def _run_full_cell(spec: Mapping[str, Any], cohort: str, arm: str,
                   seed: int, *, resume: bool) -> dict[str, Any]:
    require_valid_lock(spec)
    require_exact_gpu(spec, cohort)
    prepared_path = _receipt_path(spec, "prepare", cohort)
    if not prepared_path.is_file():
        raise SageMemRunError(f"prepare receipt missing: {cohort}")
    prepared = _load_json(prepared_path)
    validate_manifest_identity(prepared, spec, stage="prepare")
    validate_prepared_payload(spec, cohort, prepared["result"])
    final = cell_directory(spec, cohort, arm, seed)
    if resume:
        _clear_dead_partial_directories(final)
    if final.exists():
        if not resume:
            raise FileExistsError(f"formal cell already exists: {final}")
        return validate_cell_directory(spec, final, cohort, arm, seed)
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = final.parent / f".{final.name}.partial-{os.getpid()}-{uuid.uuid4().hex}"
    staging.mkdir(mode=0o750)
    try:
        model_contract, adapter = _adapter(spec, cohort)
        result = adapter.run_formal_cell(
            arm=arm, seed=seed, output_directory=staging,
            model_contract=model_contract, prepared=prepared["result"],
        )
        if not isinstance(result, Mapping):
            raise SageMemRunError("run_formal_cell must return a mapping")
        validate_formal_payload(spec, cohort, arm, seed, result)
        artifact = staging / result["episode_results"]["path"]
        if not artifact.is_file() or sha256_file(artifact) != \
                result["episode_results"]["sha256"]:
            raise SageMemRunError("formal result artifact hash mismatch")
        manifest = {
            "schema_version": 1, "study": "sage-mem-v1", "stage": "full",
            "status": "complete", "cohort": cohort, "arm": arm,
            "seed": seed, "physical_gpu": spec["cohorts"][cohort]["gpu"],
            "protocol_fingerprint": spec_fingerprint(spec),
            "result": dict(result), "completed_unix_ns": time.time_ns(),
        }
        atomic_json(staging / "manifest.json", manifest)
        os.rename(staging, final)
        return validate_cell_directory(spec, final, cohort, arm, seed)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=(
        "preflight", "smoke", "development", "seal", "prepare", "full"),
        required=True)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--cohort", choices=COHORTS)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--formal-confirmation")
    return parser.parse_args(argv)


def _validate_stage_args(args: argparse.Namespace) -> None:
    if args.stage in ("smoke", "development", "prepare", "full") \
            and args.cohort is None:
        raise SageMemRunError(f"--cohort is required for {args.stage}")
    if args.stage in ("development", "full") \
            and (args.arm is None or args.seed is None):
        raise SageMemRunError(
            f"--arm and --seed are required for {args.stage}")
    if args.stage not in ("development", "full") \
            and (args.arm is not None or args.seed is not None):
        raise SageMemRunError(
            "--arm/--seed are valid only for development or full")
    if args.stage == "development" and args.seed not in DEVELOPMENT_SEEDS:
        raise SageMemRunError(
            f"development seed must be one of {DEVELOPMENT_SEEDS}")
    if args.stage == "full" and args.seed not in FORMAL_SEEDS:
        raise SageMemRunError(f"formal seed must be one of {FORMAL_SEEDS}")


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_stage_args(args)
    spec = load_spec(args.spec, verify_parent_paths=False)
    if not args.execute:
        print(canonical_json({
            "study": "sage-mem-v1", "preview": True, "stage": args.stage,
            "cohort": args.cohort, "arm": args.arm, "seed": args.seed,
            "physical_gpu": (spec["cohorts"][args.cohort]["gpu"]
                             if args.cohort else None),
            "formal_confirmation_required": args.stage == "full",
            "formal_execution_started": False,
        }))
        return
    if args.stage == "preflight":
        result = preflight_report(spec, require_integrations=True)
        destination = _receipt_path(spec, "preflight")
    elif args.stage == "smoke":
        result = _smoke(spec, args.cohort)
        destination = _receipt_path(spec, "smoke", args.cohort)
    elif args.stage == "development":
        result = _run_development_cell(
            spec, args.cohort, args.arm, args.seed, resume=args.resume)
        print(canonical_json(result))
        return
    elif args.stage == "seal":
        result = _seal(spec)
        destination = resolve_repo_path(spec["implementation_lock"])
    elif args.stage == "prepare":
        result = _prepare(spec, args.cohort)
        destination = _receipt_path(spec, "prepare", args.cohort)
    else:
        if args.formal_confirmation != FORMAL_CONFIRMATION:
            raise SageMemRunError(
                f"full stage requires --formal-confirmation {FORMAL_CONFIRMATION}")
        result = _run_full_cell(
            spec, args.cohort, args.arm, args.seed, resume=args.resume)
        print(canonical_json(result))
        return
    if destination.exists():
        if args.resume:
            existing = _load_json(destination)
            validate_manifest_identity(existing, spec, stage=args.stage)
            print(canonical_json(existing))
            return
        raise FileExistsError(f"stage receipt already exists: {destination}")
    atomic_json(destination, result)
    print(canonical_json(result))


if __name__ == "__main__":
    main()
