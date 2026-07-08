#!/usr/bin/env python3
"""Prepare and seal all SAGE-Mem v1 formal banks without running a cell.

The preparation boundary is deliberately separate from the formal runner:

* GPU workers call only the isolated LeWM/DINO planners and materializers;
* the two DINO PushT cohorts share one lock and one joint selection plan;
* every resume rehashes the complete bank and custody artifacts;
* development selection receipts are handled as opaque file identities only;
* the final output is the generic custody registry consumed by the post-grid
  finalizer, not a semantic-label read or a formal result.

No action occurs on import.  The command-line preparation stage requires the
literal confirmation phrase ``PREPARE_SAGE_MEM_V1_FORMAL``.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_SPEC = ROOT / "configs/sage_mem_v1.yaml"
CONFIRMATION = "PREPARE_SAGE_MEM_V1_FORMAL"
PREPARATION_SCHEMA = "sage_mem_v1_formal_preparation_v1"
COHORT_RECEIPT_SCHEMA = "sage_mem_v1_formal_prepared_cohort_v1"
CUSTODY_REGISTRY_SCHEMA = "sage_mem_v1_custody_vault_registry_v1"
COHORTS = (
    "lewm_reacher_color",
    "lewm_pusht_color",
    "dinowm_pusht_token",
    "dinowm_pusht_binding",
    "dinowm_pointmaze_goal",
)
LEWM_COHORTS = COHORTS[:2]
DINO_PUSHT_COHORTS = COHORTS[2:4]
POINTMAZE_COHORT = COHORTS[4]
CLASSES = {
    "lewm_reacher_color": 4,
    "lewm_pusht_color": 4,
    "dinowm_pusht_token": 4,
    "dinowm_pusht_binding": 6,
    "dinowm_pointmaze_goal": 4,
}
WORKER_GROUPS = ("lewm", "dinowm_pusht", "dinowm_pointmaze")


class FormalPreparationError(RuntimeError):
    """A preparation, resume, or custody invariant failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FormalPreparationError(message)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _atomic_json(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(_canonical_json(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return _sha256_file(path)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"),
                           parse_constant=lambda token: (_ for _ in ()).throw(
                               FormalPreparationError(
                                   f"non-finite JSON in {label}: {token}")))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FormalPreparationError(f"cannot read {label}: {path}") from error
    _require(isinstance(value, dict), f"{label} is not a JSON mapping")
    return value


def _mode(path: Path) -> str:
    return oct(path.stat().st_mode & 0o777)


def _identity(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(),
             f"identity target is absent or a symlink: {path}")
    display: str
    if relative_to is None:
        display = str(path.resolve())
    else:
        try:
            display = str(path.resolve().relative_to(relative_to.resolve()))
        except ValueError as error:
            raise FormalPreparationError(
                f"identity target leaves preparation root: {path}") from error
    return {"path": display, "size": path.stat().st_size,
            "sha256": _sha256_file(path)}


@dataclass(frozen=True)
class PreparationLayout:
    """All durable paths owned by the preparation stage."""

    study_root: Path
    root: Path
    protocol_lock: Path
    comparator_root: Path

    @property
    def banks(self) -> Path:
        return self.root / "banks"

    @property
    def custody(self) -> Path:
        return self.root / "custody"

    @property
    def vaults(self) -> Path:
        return self.custody / "vaults"

    @property
    def custody_receipts(self) -> Path:
        return self.custody / "receipts"

    @property
    def cohort_receipts(self) -> Path:
        return self.root / "receipts"

    @property
    def locks(self) -> Path:
        return self.root / "locks"

    @property
    def registry(self) -> Path:
        return self.custody / "registry.json"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    def bank(self, cohort: str) -> Path:
        return self.banks / cohort

    def vault(self, cohort: str) -> Path:
        return self.vaults / f"{cohort}.npz"

    def custody_receipt(self, cohort: str) -> Path:
        return self.custody_receipts / f"{cohort}.json"

    def cohort_receipt(self, cohort: str) -> Path:
        return self.cohort_receipts / f"{cohort}.json"

    def comparator_receipt(self, cohort: str) -> Path:
        return self.comparator_root / cohort / "receipt.json"

    def ensure_directories(self) -> None:
        for path in (self.banks, self.vaults, self.custody_receipts,
                     self.cohort_receipts, self.locks):
            path.mkdir(parents=True, exist_ok=True)


def layout_from_spec(
        spec: Mapping[str, Any], *, preparation_root: Path | None = None,
        comparator_root: Path | None = None,
        protocol_lock: Path | None = None) -> PreparationLayout:
    execution = spec.get("execution")
    _require(isinstance(execution, Mapping)
             and isinstance(execution.get("output_root"), str),
             "SAGE-Mem output root is missing")
    study_root = (ROOT / execution["output_root"]).resolve()
    lock_value = spec.get("implementation_lock")
    _require(isinstance(lock_value, str) and lock_value,
             "SAGE-Mem implementation lock path is missing")
    return PreparationLayout(
        study_root=study_root,
        root=(Path(preparation_root).resolve() if preparation_root is not None
              else study_root / "formal_preparation"),
        protocol_lock=(Path(protocol_lock).resolve()
                       if protocol_lock is not None
                       else (ROOT / lock_value).resolve()),
        comparator_root=(Path(comparator_root).resolve()
                         if comparator_root is not None
                         else study_root / "development" / "selections"),
    )


def _spec_identity(spec: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(spec.get("_spec_path", ""))).resolve()
    expected = spec.get("_spec_sha256")
    _require(path.is_file() and _is_sha256(expected)
             and _sha256_file(path) == expected,
             "loaded SAGE-Mem specification identity changed")
    return _identity(path)


def _opaque_boundary_identities(
        spec: Mapping[str, Any], cohort: str,
        layout: PreparationLayout) -> dict[str, Any]:
    """Hash locked boundaries without parsing development outcome contents."""

    _require(cohort in COHORTS, f"unknown cohort: {cohort}")
    comparator = layout.comparator_receipt(cohort)
    _require(layout.protocol_lock.is_file(),
             "formal protocol lock is absent")
    _require(comparator.is_file(),
             f"locked comparator receipt is absent: {cohort}")
    return {
        "study_protocol": _spec_identity(spec),
        "implementation_lock": _identity(layout.protocol_lock),
        # Deliberately opaque: no JSON parser is called on this file.
        "locked_comparator_receipt": _identity(comparator),
    }


@contextmanager
def _exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.seek(0)
        stream.truncate()
        stream.write(f"pid={os.getpid()}\n")
        stream.flush()
        os.fsync(stream.fileno())
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _reject_partial_paths(layout: PreparationLayout) -> None:
    if not layout.root.exists():
        return
    partial = sorted(path for path in layout.root.rglob("*")
                     if ".partial-" in path.name or path.name.endswith(".tmp"))
    _require(not partial,
             f"partial formal-preparation artifacts exist: {partial}")


def _reject_cohort_partial_paths(
        layout: PreparationLayout, cohort: str) -> None:
    """Reject dead partials for one cohort without racing another GPU worker."""

    candidates = [
        *layout.banks.glob(f".{cohort}.partial-*"),
        *layout.vaults.glob(f".{cohort}.npz.*.tmp"),
        *layout.custody_receipts.glob(f".{cohort}.json.*.tmp"),
        *layout.cohort_receipts.glob(f".{cohort}.json.*.tmp"),
    ]
    _require(not candidates,
             f"partial formal-preparation artifacts exist for {cohort}: "
             f"{sorted(candidates)}")


def _seal_bank_tree(root: Path) -> None:
    _require(root.is_dir() and not root.is_symlink(),
             f"formal bank root is absent or unsafe: {root}")
    paths = sorted(root.rglob("*"), key=lambda value: len(value.parts),
                   reverse=True)
    for path in paths:
        _require(not path.is_symlink(), f"formal bank contains symlink: {path}")
        os.chmod(path, 0o555 if path.is_dir() else 0o444)
    os.chmod(root, 0o555)


def _assert_bank_tree_immutable(root: Path) -> None:
    _require(root.is_dir() and not root.is_symlink(),
             f"formal bank root is absent or unsafe: {root}")
    for path in (root, *root.rglob("*")):
        _require(not path.is_symlink(), f"formal bank contains symlink: {path}")
        _require((path.stat().st_mode & 0o222) == 0,
                 f"formal bank remains writable: {path}")


@dataclass(frozen=True)
class PreparedEvidence:
    cohort: str
    bank_manifest: Mapping[str, Any]
    custody_receipt: Mapping[str, Any]
    vault_sha256: str
    custody_record: Mapping[str, Any]
    backend_proof: Mapping[str, Any]


def _validate_dino_custody(
        *, cohort: str, plan: Any, layout: PreparationLayout
        ) -> PreparedEvidence:
    from scripts.sage_mem_v1_dino_formal import (
        validate_materialized_bank_provenance,
    )

    bank_root = layout.bank(cohort)
    manifest_path = bank_root / "manifest.json"
    provenance = validate_materialized_bank_provenance(bank_root)
    manifest = _read_json(manifest_path, f"{cohort} DINO bank manifest")
    _require(manifest.get("plan_sha256") == plan.plan_sha256,
             f"{cohort} bank does not match the joint sealed plan")
    receipt_path = layout.custody_receipt(cohort)
    vault = layout.vault(cohort)
    _require(receipt_path.is_file() and not receipt_path.is_symlink()
             and _mode(receipt_path) == "0o400",
             f"{cohort} DINO custody receipt is not sealed")
    custody = _read_json(receipt_path, f"{cohort} DINO custody receipt")
    required = {
        "schema", "api_version", "status", "cohort", "plan_sha256",
        "path", "size", "sha256", "mode", "per_cell_api_access",
    }
    _require(set(custody) == required
             and custody.get("schema") ==
             "sage_mem_v1_dino_formal_label_custody_v1"
             and custody.get("status") == "sealed-for-post-grid-finalizer"
             and custody.get("cohort") == cohort
             and custody.get("plan_sha256") == plan.plan_sha256
             and custody.get("per_cell_api_access") is False,
             f"{cohort} DINO custody schema/identity changed")
    _require(vault.is_file() and not vault.is_symlink()
             and Path(str(custody.get("path", ""))).resolve() == vault.resolve()
             and custody.get("size") == vault.stat().st_size
             and custody.get("sha256") == _sha256_file(vault)
             and custody.get("sha256") == manifest.get(
                 "sealed_label_vault_sha256")
             and custody.get("mode") == "0o400" and _mode(vault) == "0o400",
             f"{cohort} DINO vault identity changed")
    try:
        relative_vault = vault.resolve().relative_to(layout.custody.resolve())
    except ValueError as error:
        raise FormalPreparationError(
            f"{cohort} DINO vault leaves custody root") from error
    artifact = {
        "path": str(relative_vault),
        "sha256": custody["sha256"],
        "size": custody["size"],
    }
    source = {
        "artifact": artifact,
        "keys": {
            "episode_id": "episode_id",
            "native_cluster_id": "native_cluster_id",
            "label": "class_id",
        },
    }
    bank_identity = _identity(manifest_path, relative_to=layout.root)
    return PreparedEvidence(
        cohort=cohort,
        bank_manifest=bank_identity,
        custody_receipt=_identity(receipt_path, relative_to=layout.root),
        vault_sha256=str(custody["sha256"]),
        custody_record={
            "bank_manifest_sha256": bank_identity["sha256"],
            "classes": CLASSES[cohort],
            "sources": {
                "formal_test": source,
                "consumer_train": source,
            },
        },
        backend_proof={
            "backend": "DINO-WM",
            "plan_sha256": plan.plan_sha256,
            "provenance_manifest_sha256": provenance["manifest_sha256"],
            "parent_episode_overlap_count": provenance[
                "freshness_proof"]["parent_episode_overlap_count"],
            "cross_split_native_episode_overlap_count": provenance[
                "freshness_proof"][
                    "cross_split_native_episode_overlap_count"],
        },
    )


def _validate_lewm_custody(
        *, cohort: str, spec: Mapping[str, Any], layout: PreparationLayout
        ) -> PreparedEvidence:
    from scripts.sage_mem_v1_lewm_formal import (
        finalizer_custody_record,
        sealed_label_vault_handle,
        validate_lewm_formal_manifest,
    )

    manifest_path = layout.bank(cohort) / "manifest.json"
    counts = {
        split: int(spec["cohorts"][cohort]["split_episodes"][split])
        for split in ("formal_train", "consumer_train", "formal_test")
    }
    manifest = validate_lewm_formal_manifest(
        manifest_path, expected_cohort=cohort, expected_counts=counts)
    receipt_path = layout.custody_receipt(cohort)
    sealed = sealed_label_vault_handle(manifest_path, receipt_path)
    record = finalizer_custody_record(
        manifest_path, receipt_path, registry_root=layout.custody)
    bank_identity = _identity(manifest_path, relative_to=layout.root)
    _require(record["bank_manifest_sha256"] == bank_identity["sha256"]
             and record["classes"] == CLASSES[cohort],
             f"{cohort} LeWM finalizer custody binding changed")
    return PreparedEvidence(
        cohort=cohort,
        bank_manifest=bank_identity,
        custody_receipt=_identity(receipt_path, relative_to=layout.root),
        vault_sha256=str(sealed["vault_sha256"]),
        custody_record=record,
        backend_proof={
            "backend": "SIGReg-LeWM",
            "host_hash_before": manifest["host_hash_before"],
            "host_hash_after": manifest["host_hash_after"],
            "parent_overlap_zero": manifest["admissions"][
                "parent_overlap_zero"],
            "formal_split_overlap_zero": manifest["admissions"][
                "formal_split_overlap_zero"],
        },
    )


def _cohort_receipt_value(
        spec: Mapping[str, Any], layout: PreparationLayout,
        evidence: PreparedEvidence) -> dict[str, Any]:
    return {
        "schema": COHORT_RECEIPT_SCHEMA,
        "study": "sage-mem-v1",
        "status": "prepared-and-hash-validated",
        "cohort": evidence.cohort,
        "formal_jobs_launched": False,
        "formal_outcomes_read": False,
        "development_outcomes_read": False,
        "development_access": "opaque locked comparator receipt identity only",
        "boundaries": _opaque_boundary_identities(
            spec, evidence.cohort, layout),
        "bank_manifest": dict(evidence.bank_manifest),
        "custody_receipt": dict(evidence.custody_receipt),
        "sealed_vault_sha256": evidence.vault_sha256,
        "backend_proof": dict(evidence.backend_proof),
        "finalizer_custody_record": dict(evidence.custody_record),
    }


def _publish_or_validate_cohort_receipt(
        spec: Mapping[str, Any], layout: PreparationLayout,
        evidence: PreparedEvidence, *, allow_create: bool = True
        ) -> dict[str, Any]:
    expected = _cohort_receipt_value(spec, layout, evidence)
    path = layout.cohort_receipt(evidence.cohort)
    if path.exists():
        _require(path.is_file() and not path.is_symlink()
                 and _mode(path) == "0o444",
                 f"cohort receipt is not immutable: {path}")
        _require(_read_json(path, "cohort preparation receipt") == expected,
                 f"cohort preparation receipt differs: {evidence.cohort}")
    else:
        _require(allow_create,
                 f"cohort receipt is absent in validation mode: "
                 f"{evidence.cohort}")
        _atomic_json(path, expected)
        os.chmod(path, 0o444)
    return expected


def _ensure_cohort(
        *, spec: Mapping[str, Any], cohort: str,
        layout: PreparationLayout, materialize: Callable[[], None],
        validate: Callable[[], PreparedEvidence]) -> PreparedEvidence:
    """Create or fully revalidate one three-artifact bank/custody unit."""

    _require(cohort in COHORTS, f"unknown preparation cohort: {cohort}")
    layout.ensure_directories()
    _reject_cohort_partial_paths(layout, cohort)
    bank = layout.bank(cohort)
    vault = layout.vault(cohort)
    custody = layout.custody_receipt(cohort)
    receipt = layout.cohort_receipt(cohort)
    core = (bank.exists(), vault.exists(), custody.exists())
    if any(core) and not all(core):
        raise FormalPreparationError(
            f"partial formal bank/custody state for {cohort}: {core}")
    if not any(core):
        _require(not receipt.exists(),
                 f"receipt exists before bank materialization: {cohort}")
        _require(not layout.registry.exists() and not layout.manifest.exists(),
                 "cannot add a cohort after the custody registry was sealed")
        materialize()
        _require(bank.exists() and vault.is_file() and custody.is_file(),
                 f"materializer did not publish complete core: {cohort}")
    _seal_bank_tree(bank)
    evidence = validate()
    _require(evidence.cohort == cohort,
             f"backend validation returned another cohort: {cohort}")
    _publish_or_validate_cohort_receipt(spec, layout, evidence)
    return evidence


def _materialize_lewm(
        spec: Mapping[str, Any], cohort: str, layout: PreparationLayout) -> None:
    from scripts.sage_mem_v1_lewm_formal import prepare_lewm_formal_bank

    prepare_lewm_formal_bank(
        cohort=cohort,
        spec=spec,
        output_directory=layout.bank(cohort),
        label_vault_path=layout.vault(cohort),
        label_custody_receipt_path=layout.custody_receipt(cohort),
    )


def _materialize_dino(
        plan: Any, cohort: str, layout: PreparationLayout) -> None:
    from scripts.sage_mem_v1_dino_formal import materialize_dino_formal_bank

    materialize_dino_formal_bank(
        plan,
        layout.bank(cohort),
        label_vault_destination=layout.vault(cohort),
        label_vault_receipt_destination=layout.custody_receipt(cohort),
    )


def _release_worker_memory() -> None:
    """Release one cohort's host before opening the next on the same GPU."""

    gc.collect()
    try:
        import torch
    except ModuleNotFoundError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def prepare_lewm_group(
        spec: Mapping[str, Any], layout: PreparationLayout, *,
        materializer: Callable[[Mapping[str, Any], str,
                                PreparationLayout], None] = _materialize_lewm,
        validator: Callable[..., PreparedEvidence] = _validate_lewm_custody,
        ) -> dict[str, PreparedEvidence]:
    with _exclusive_lock(layout.locks / "lewm.lock"):
        result: dict[str, PreparedEvidence] = {}
        for cohort in LEWM_COHORTS:
            result[cohort] = _ensure_cohort(
                spec=spec, cohort=cohort, layout=layout,
                materialize=lambda cohort=cohort: materializer(
                    spec, cohort, layout),
                validate=lambda cohort=cohort: validator(
                    cohort=cohort, spec=spec, layout=layout),
            )
            _release_worker_memory()
        return result


def prepare_dino_pusht_group(
        spec: Mapping[str, Any], layout: PreparationLayout, *,
        planner: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
        materializer: Callable[[Any, str, PreparationLayout], None]
        = _materialize_dino,
        validator: Callable[..., PreparedEvidence] = _validate_dino_custody,
        ) -> dict[str, PreparedEvidence]:
    """Prepare both PushT banks under one lock and one joint plan."""

    if planner is None:
        from scripts.sage_mem_v1_dino_formal import (
            plan_pusht_formal_pair_from_spec,
        )
        planner = plan_pusht_formal_pair_from_spec
    with _exclusive_lock(layout.locks / "dinowm_pusht_joint.lock"):
        plans = planner(spec)
        _require(set(plans) == set(DINO_PUSHT_COHORTS),
                 "joint DINO PushT planner returned another cohort registry")
        left = plans[DINO_PUSHT_COHORTS[0]].native_episodes
        right = plans[DINO_PUSHT_COHORTS[1]].native_episodes
        _require(not set(left).intersection(right),
                 "joint DINO PushT plans overlap native episodes")
        result: dict[str, PreparedEvidence] = {}
        for cohort in DINO_PUSHT_COHORTS:
            plan = plans[cohort]
            result[cohort] = _ensure_cohort(
                spec=spec, cohort=cohort, layout=layout,
                materialize=lambda plan=plan, cohort=cohort: materializer(
                    plan, cohort, layout),
                validate=lambda plan=plan, cohort=cohort: validator(
                    cohort=cohort, plan=plan, layout=layout),
            )
            _release_worker_memory()
        return result


def prepare_dino_pointmaze_group(
        spec: Mapping[str, Any], layout: PreparationLayout, *,
        planner: Callable[[Mapping[str, Any]], Any] | None = None,
        materializer: Callable[[Any, str, PreparationLayout], None]
        = _materialize_dino,
        validator: Callable[..., PreparedEvidence] = _validate_dino_custody,
        ) -> dict[str, PreparedEvidence]:
    if planner is None:
        from scripts.sage_mem_v1_dino_formal import (
            plan_pointmaze_formal_from_spec,
        )
        planner = plan_pointmaze_formal_from_spec
    with _exclusive_lock(layout.locks / "dinowm_pointmaze.lock"):
        plan = planner(spec)
        cohort = POINTMAZE_COHORT
        return {cohort: _ensure_cohort(
            spec=spec, cohort=cohort, layout=layout,
            materialize=lambda: materializer(plan, cohort, layout),
            validate=lambda: validator(
                cohort=cohort, plan=plan, layout=layout),
        )}


def _collect_all_evidence(
        spec: Mapping[str, Any], layout: PreparationLayout
        ) -> dict[str, PreparedEvidence]:
    from scripts.sage_mem_v1_dino_formal import (
        plan_pointmaze_formal_from_spec,
        plan_pusht_formal_pair_from_spec,
    )

    push = plan_pusht_formal_pair_from_spec(spec)
    point = plan_pointmaze_formal_from_spec(spec)
    result = {
        cohort: _validate_lewm_custody(
            cohort=cohort, spec=spec, layout=layout)
        for cohort in LEWM_COHORTS
    }
    result.update({
        cohort: _validate_dino_custody(
            cohort=cohort, plan=push[cohort], layout=layout)
        for cohort in DINO_PUSHT_COHORTS
    })
    result[POINTMAZE_COHORT] = _validate_dino_custody(
        cohort=POINTMAZE_COHORT, plan=point, layout=layout)
    return result


def _registry_value(
        evidence: Mapping[str, PreparedEvidence]) -> dict[str, Any]:
    _require(set(evidence) == set(COHORTS),
             "custody registry requires all five cohorts")
    return {
        "schema": CUSTODY_REGISTRY_SCHEMA,
        "study": "sage-mem-v1",
        "status": "sealed",
        "labels_available_only_after_complete_phase_a_grid": True,
        "development_outcomes_read": False,
        "cohorts": {
            cohort: dict(evidence[cohort].custody_record)
            for cohort in COHORTS
        },
    }


def publish_custody_registry(
        spec: Mapping[str, Any], layout: PreparationLayout, *,
        evidence_loader: Callable[[Mapping[str, Any], PreparationLayout],
                                  Mapping[str, PreparedEvidence]]
        = _collect_all_evidence,
        allow_create: bool = True) -> dict[str, Any]:
    """Validate every bank/vault and atomically seal the generic registry."""

    layout.ensure_directories()
    with _exclusive_lock(layout.locks / "custody_registry.lock"):
        _reject_partial_paths(layout)
        evidence = dict(evidence_loader(spec, layout))
        _require(set(evidence) == set(COHORTS),
                 "validated evidence does not cover all five cohorts")
        for cohort in COHORTS:
            _assert_bank_tree_immutable(layout.bank(cohort))
            _publish_or_validate_cohort_receipt(
                spec, layout, evidence[cohort], allow_create=allow_create)
        registry = _registry_value(evidence)
        if layout.registry.exists():
            _require(layout.registry.is_file()
                     and not layout.registry.is_symlink()
                     and _mode(layout.registry) == "0o400",
                     "custody registry is not immutable")
            _require(_read_json(layout.registry, "custody registry") == registry,
                     "sealed custody registry differs from validated banks")
        else:
            _require(allow_create,
                     "custody registry is absent during validation-only mode")
            _atomic_json(layout.registry, registry)
            os.chmod(layout.registry, 0o400)
        cohort_receipts = {
            cohort: _identity(
                layout.cohort_receipt(cohort), relative_to=layout.root)
            for cohort in COHORTS
        }
        manifest = {
            "schema": PREPARATION_SCHEMA,
            "study": "sage-mem-v1",
            "status": "complete",
            "formal_jobs_launched": False,
            "formal_outcomes_read": False,
            "development_outcomes_read": False,
            "development_access":
                "opaque locked comparator receipt identities only",
            "study_protocol": _spec_identity(spec),
            "implementation_lock": _identity(layout.protocol_lock),
            "custody_registry": _identity(
                layout.registry, relative_to=layout.root),
            "cohort_receipts": cohort_receipts,
        }
        if layout.manifest.exists():
            _require(layout.manifest.is_file()
                     and not layout.manifest.is_symlink()
                     and _mode(layout.manifest) == "0o444",
                     "formal preparation manifest is not immutable")
            _require(_read_json(layout.manifest,
                                "formal preparation manifest") == manifest,
                     "formal preparation manifest differs")
        else:
            _require(allow_create,
                     "formal preparation manifest is absent in validate mode")
            _atomic_json(layout.manifest, manifest)
            os.chmod(layout.manifest, 0o444)
        return manifest


def _isolated_dino_python(spec: Mapping[str, Any], cohort: str) -> Path:
    parent = ROOT / spec["cohorts"][cohort]["parent_protocol"]
    cfg = yaml.safe_load(parent.read_text())
    value = cfg.get("execution", {}).get("isolated_python")
    _require(isinstance(value, str) and value,
             f"DINO isolated Python is missing: {cohort}")
    path = ROOT / value
    _require(path.is_file(), f"DINO isolated Python is unavailable: {path}")
    return path


def worker_commands(
        spec: Mapping[str, Any], spec_path: Path,
        layout: PreparationLayout) -> tuple[tuple[list[str], dict[str, str]], ...]:
    """Return preparation-only worker commands and their isolated GPU envs."""

    script = Path(__file__).resolve()
    common = ["--stage", "worker", "--spec", str(spec_path.resolve()),
              "--preparation-root", str(layout.root),
              "--comparator-root", str(layout.comparator_root),
              "--protocol-lock", str(layout.protocol_lock),
              "--confirmation", CONFIRMATION]
    lewm_python = ROOT / ".venv" / "bin" / "python"
    if not lewm_python.is_file():
        lewm_python = Path(sys.executable)
    definitions = (
        ("lewm", lewm_python, "0"),
        ("dinowm_pusht", _isolated_dino_python(
            spec, DINO_PUSHT_COHORTS[0]), "1"),
        ("dinowm_pointmaze", _isolated_dino_python(
            spec, POINTMAZE_COHORT), "2"),
    )
    result = []
    for group, python, gpu in definitions:
        command = [str(python), str(script), *common, "--worker", group]
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        environment["PYTHONUNBUFFERED"] = "1"
        result.append((command, environment))
    return tuple(result)


def coordinate_preparation(
        spec: Mapping[str, Any], spec_path: Path,
        layout: PreparationLayout, *, confirmation: str) -> dict[str, Any]:
    _require(confirmation == CONFIRMATION,
             "formal preparation confirmation phrase is absent")
    layout.ensure_directories()
    processes: list[subprocess.Popen[Any]] = []
    try:
        for command, environment in worker_commands(
                spec, spec_path, layout):
            processes.append(subprocess.Popen(
                command, cwd=ROOT, env=environment))
        failed: list[int] = []
        for process in processes:
            code = process.wait()
            if code != 0:
                failed.append(code)
        if failed:
            raise FormalPreparationError(
                f"formal preparation worker failed: {failed}")
    except BaseException:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            if process.poll() is None:
                process.wait()
        raise
    return publish_custody_registry(spec, layout)


def _load_spec(path: Path) -> dict[str, Any]:
    from scripts.sage_mem_v1_spec import load_spec
    return load_spec(path, verify_parent_paths=True)


def _run_worker(group: str, spec: Mapping[str, Any],
                layout: PreparationLayout) -> None:
    if group == "lewm":
        prepare_lewm_group(spec, layout)
    elif group == "dinowm_pusht":
        prepare_dino_pusht_group(spec, layout)
    elif group == "dinowm_pointmaze":
        prepare_dino_pointmaze_group(spec, layout)
    else:
        raise FormalPreparationError(f"unknown worker group: {group}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "worker", "validate"),
                        required=True)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--preparation-root", type=Path)
    parser.add_argument("--comparator-root", type=Path)
    parser.add_argument("--protocol-lock", type=Path)
    parser.add_argument("--confirmation")
    parser.add_argument("--worker", choices=WORKER_GROUPS)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    spec = _load_spec(args.spec)
    layout = layout_from_spec(
        spec, preparation_root=args.preparation_root,
        comparator_root=args.comparator_root,
        protocol_lock=args.protocol_lock)
    if args.stage == "validate":
        publish_custody_registry(
            spec, layout, allow_create=False)
        return 0
    _require(args.confirmation == CONFIRMATION,
             "formal preparation confirmation phrase is absent")
    if args.stage == "worker":
        _require(args.worker is not None,
                 "worker stage requires --worker")
        _run_worker(args.worker, spec, layout)
        return 0
    _require(args.worker is None,
             "coordinator prepare stage does not accept --worker")
    coordinate_preparation(
        spec, args.spec, layout, confirmation=args.confirmation)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FormalPreparationError as error:
        print(f"formal preparation stopped: {error}", file=sys.stderr)
        raise SystemExit(2) from error


__all__ = [
    "CONFIRMATION", "PREPARATION_SCHEMA", "COHORT_RECEIPT_SCHEMA",
    "CUSTODY_REGISTRY_SCHEMA", "COHORTS", "LEWM_COHORTS",
    "DINO_PUSHT_COHORTS", "POINTMAZE_COHORT", "CLASSES",
    "FormalPreparationError", "PreparationLayout", "PreparedEvidence",
    "layout_from_spec", "prepare_lewm_group", "prepare_dino_pusht_group",
    "prepare_dino_pointmaze_group", "publish_custody_registry",
    "worker_commands", "coordinate_preparation", "parse_args", "main",
]
