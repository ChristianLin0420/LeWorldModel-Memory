#!/usr/bin/env python3
"""Outcome-blind, crash-safe closeout for the sealed SAGE-Mem v1 grid.

The experiment producer and formal finalizer are protocol locked.  This
unsealed reliability wrapper does not change either one.  Before any label
reveal it seals the complete label-free execution-deck tree, the label-custody
registry identity, the raw-context producer summary, and the protocol lock in
a durable custody receipt.  It also installs an empty read-only output-root
interlock so that the ordinary launcher cannot bypass this pre-reveal check.

After all 600 Phase-A cells exist and no campaign pipeline process remains,
``--recover`` takes the campaign lock.  A missing reveal receipt is safe only
for an absent or empty pre-reveal root and starts the unchanged finalizer from
scratch.  A valid durable receipt authorizes recovery of a partial root: the
partial tree is archived, the receipt is copied byte-for-byte into fresh
staging through a narrowly replaced receipt-writer callback, the unchanged
sealed finalizer reconstructs the output, its own validator authenticates the
staging tree, and an atomic rename installs it.  An invalid receipt is never
archived or retried.

The resulting receipts are operational provenance only.  They do not inspect
effect directions, pass/fail gates, or authorize any manuscript claim.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time
from types import ModuleType
from typing import Any, Callable, Iterable, Mapping, Sequence
import uuid


ROOT = Path(__file__).resolve().parents[1]
STUDY_ROOT = ROOT / "outputs/sage_mem_v1"
SPEC_PATH = ROOT / "configs/sage_mem_v1.yaml"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PRE_REVEAL_SCHEMA = "sage_mem_v1_closeout_custody_v1"
RECOVERY_SCHEMA = "sage_mem_v1_closeout_recovery_v1"
STATE_SCHEMA = "sage_mem_v1_closeout_state_v1"
STUDY = "sage-mem-v1"
EXPECTED_CELLS = 600

FINALIZER_RELATIVE = "scripts/sage_mem_v1_formal_finalizer.py"
AUDIT_RELATIVE = "scripts/audit_sage_mem_v1.py"
FORMAL_AUDIT_RELATIVE = "scripts/audit_sage_mem_v1_formal.py"
SPEC_RELATIVE = "configs/sage_mem_v1.yaml"
RECOVERY_RELATIVE = "scripts/recover_sage_mem_v1_closeout.py"
WATCHDOG_RELATIVE = "scripts/watch_sage_mem_v1_campaign.py"

PIPELINE_SCRIPTS = {
    "scripts/run_sage_mem_v1_campaign.py",
    "scripts/launch_sage_mem_v1.py",
    "scripts/run_sage_mem_v1.py",
    FINALIZER_RELATIVE,
    AUDIT_RELATIVE,
    FORMAL_AUDIT_RELATIVE,
    RECOVERY_RELATIVE,
}


class CloseoutRecoveryError(RuntimeError):
    """The closeout cannot continue without weakening a sealed boundary."""


class CloseoutStopRequested(CloseoutRecoveryError):
    """A stop sentinel or signal requested a clean recovery interruption."""


@dataclass(frozen=True)
class CloseoutPaths:
    repo_root: Path
    study_root: Path
    spec_path: Path
    phase_a_root: Path
    final_root: Path
    protocol_lock: Path
    label_registry: Path
    raw_context_root: Path
    execution_root: Path
    execution_registry: Path
    receipt_root: Path
    pre_reveal_receipt: Path
    reveal_anchor: Path
    recovery_receipt: Path
    state_receipt: Path
    archive_root: Path
    campaign_lock: Path
    report_path: Path

    @classmethod
    def production(cls, repo_root: Path = ROOT,
                   study_root: Path = STUDY_ROOT,
                   spec_path: Path = SPEC_PATH) -> "CloseoutPaths":
        repo = Path(os.path.abspath(repo_root))
        study = Path(os.path.abspath(study_root))
        spec = Path(os.path.abspath(spec_path))
        preparation = study / "formal_preparation"
        receipts = study / "receipts" / "closeout"
        execution = preparation / "execution_decks"
        return cls(
            repo_root=repo,
            study_root=study,
            spec_path=spec,
            phase_a_root=study,
            final_root=study / "formal_finalized",
            protocol_lock=study / "protocol_lock.json",
            label_registry=preparation / "custody" / "registry.json",
            raw_context_root=study / "raw_context_phase_a",
            execution_root=execution,
            execution_registry=execution / "registry.json",
            receipt_root=receipts,
            pre_reveal_receipt=receipts / "pre_reveal_custody.json",
            reveal_anchor=receipts / "label_reveal_receipt.original.json",
            recovery_receipt=receipts / "recovery_receipt.json",
            state_receipt=receipts / "recovery_state.json",
            archive_root=study / "closeout_recovery_archive",
            campaign_lock=study / "campaign.lock",
            report_path=study / "formal_audit" / "report.json",
        )


def validate_closeout_paths(paths: CloseoutPaths) -> None:
    """Require the exact lexical in-workspace production layout."""

    expected = CloseoutPaths.production(
        paths.repo_root, paths.study_root, paths.spec_path)
    for name in CloseoutPaths.__dataclass_fields__:
        observed = Path(os.path.abspath(getattr(paths, name)))
        wanted = Path(os.path.abspath(getattr(expected, name)))
        if observed != wanted:
            raise CloseoutRecoveryError(
                f"closeout path differs from registered layout: {name}")
    repo = Path(os.path.abspath(paths.repo_root))
    study = Path(os.path.abspath(paths.study_root))
    try:
        study.relative_to(repo)
    except ValueError as error:
        raise CloseoutRecoveryError(
            "study root leaves the repository workspace") from error
    for path in (
            repo, study, paths.spec_path, paths.phase_a_root,
            paths.final_root.parent, paths.protocol_lock,
            paths.label_registry, paths.raw_context_root,
            paths.execution_root, paths.execution_registry,
            paths.receipt_root, paths.pre_reveal_receipt,
            paths.reveal_anchor, paths.recovery_receipt,
            paths.state_receipt, paths.archive_root,
            paths.campaign_lock, paths.report_path.parent):
        _reject_symlink_components(path)


def _validate_stop_sentinel(paths: CloseoutPaths,
                            stop_sentinel: Path | None) -> None:
    if stop_sentinel is None:
        return
    stop = Path(os.path.abspath(stop_sentinel))
    try:
        stop.relative_to(paths.study_root)
    except ValueError as error:
        raise CloseoutRecoveryError(
            "stop sentinel leaves the study root") from error
    _reject_symlink_components(stop)


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False) + "\n").encode("utf-8")


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _reject_symlink_components(path: Path) -> None:
    """Reject a symlink in any existing component of an absolute path."""

    lexical = Path(os.path.abspath(path))
    chain = list(reversed(lexical.parents)) + [lexical]
    for component in chain:
        try:
            if component.is_symlink():
                raise CloseoutRecoveryError(
                    f"path contains a symlink component: {component}")
        except OSError as error:
            raise CloseoutRecoveryError(
                f"cannot inspect path component: {component}") from error


def _stable_file_snapshot(
        path: Path, *, hash_content: bool,
        capture_payload: bool = False) -> tuple[dict[str, Any], bytes | None]:
    """Read one regular file through one FD and revalidate its path identity."""

    path = Path(path)
    _reject_symlink_components(path)
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise CloseoutRecoveryError(f"cannot open identity file: {path}") \
            from error
    digest = hashlib.sha256() if hash_content else None
    chunks: list[bytes] | None = [] if capture_payload else None
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CloseoutRecoveryError(
                f"identity path is not a regular file: {path}")
        if digest is not None or chunks is not None:
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                if digest is not None:
                    digest.update(chunk)
                if chunks is not None:
                    chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(after, field)
           for field in identity_fields):
        raise CloseoutRecoveryError(f"identity file changed while read: {path}")
    try:
        lexical = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise CloseoutRecoveryError(
            f"identity file disappeared after read: {path}") from error
    if any(getattr(after, field) != getattr(lexical, field)
           for field in identity_fields):
        raise CloseoutRecoveryError(
            f"identity file path changed while read: {path}")
    record = {
        "mode": stat.S_IMODE(after.st_mode),
        "size": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "device": after.st_dev,
        "inode": after.st_ino,
    }
    if digest is not None:
        record["sha256"] = digest.hexdigest()
    return record, (b"".join(chunks) if chunks is not None else None)


def _sha256_file(path: Path) -> str:
    record, _ = _stable_file_snapshot(path, hash_content=True)
    return str(record["sha256"])


def _regular_file_handle(path: Path) -> dict[str, Any]:
    path = Path(path)
    record, _ = _stable_file_snapshot(path, hash_content=True)
    return {
        "path": str(path.resolve()),
        "sha256": record["sha256"],
        "size": record["size"],
    }


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise CloseoutRecoveryError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _parse_json_bytes(payload: bytes, label: str, *,
                      require_canonical: bool) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8", errors="strict")
        value = json.loads(
            text, object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token: {token}")))
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise CloseoutRecoveryError(f"cannot parse {label}") from error
    if not isinstance(value, dict):
        raise CloseoutRecoveryError(f"{label} is not a JSON mapping")
    if require_canonical and payload != _canonical_bytes(value):
        raise CloseoutRecoveryError(f"{label} is not canonical JSON")
    return value


def _read_json(path: Path, label: str, *,
               require_canonical: bool = False) -> dict[str, Any]:
    _, payload = _stable_file_snapshot(
        path, hash_content=False, capture_payload=True)
    assert payload is not None
    return _parse_json_bytes(
        payload, label, require_canonical=require_canonical)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory_tree(root: Path) -> None:
    directories = [root, *(path for path in root.rglob("*") if path.is_dir())]
    for directory in sorted(
            directories, key=lambda path: len(path.parts), reverse=True):
        if directory.is_symlink():
            raise CloseoutRecoveryError(
                f"cannot fsync symlinked directory: {directory}")
        _fsync_directory(directory)


def _atomic_write_bytes(path: Path, payload: bytes, *, exclusive: bool,
                        mode: int = 0o440) -> None:
    path = Path(path)
    _reject_symlink_components(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise CloseoutRecoveryError(f"unsafe receipt parent: {path.parent}")
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}-{uuid.uuid4().hex}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, mode)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise CloseoutRecoveryError(f"cannot write receipt: {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        if exclusive:
            try:
                os.link(temporary, path, follow_symlinks=False)
            except FileExistsError as error:
                raise CloseoutRecoveryError(
                    f"receipt already exists: {path}") from error
            temporary.unlink()
        else:
            os.replace(temporary, path)
        os.chmod(path, mode)
        _fsync_directory(path.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, value: Mapping[str, Any], *, exclusive: bool,
                 mode: int = 0o440) -> None:
    _atomic_write_bytes(
        path, _canonical_bytes(value), exclusive=exclusive, mode=mode)


def _directory_snapshot(path: Path) -> tuple[int, int, int, int]:
    _reject_symlink_components(path)
    try:
        info = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise CloseoutRecoveryError(
            f"cannot inspect identity directory: {path}") from error
    if not stat.S_ISDIR(info.st_mode):
        raise CloseoutRecoveryError(
            f"identity path is not a directory: {path}")
    return (info.st_dev, info.st_ino, stat.S_IMODE(info.st_mode),
            info.st_mtime_ns)


def _tree_inventory(root: Path, *, hash_content: bool = True,
                    require_canonical_json: bool = False) \
        -> list[dict[str, Any]]:
    root = Path(root)
    _reject_symlink_components(root)
    if not root.is_dir() or root.is_symlink():
        raise CloseoutRecoveryError(f"identity tree is missing or unsafe: {root}")
    rows: list[dict[str, Any]] = []
    directory_snapshots = {root: _directory_snapshot(root)}
    for item in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        if item.is_symlink():
            raise CloseoutRecoveryError(f"identity tree contains symlink: {item}")
        relative = item.relative_to(root).as_posix()
        if item.is_dir():
            directory = _directory_snapshot(item)
            directory_snapshots[item] = directory
            rows.append({"path": relative + "/", "type": "directory",
                         "device": directory[0], "inode": directory[1],
                         "mode": directory[2], "mtime_ns": directory[3]})
        elif item.is_file():
            is_json = item.suffix == ".json"
            snapshot, payload = _stable_file_snapshot(
                item, hash_content=hash_content,
                capture_payload=is_json)
            row = {
                "path": relative, "type": "file",
                "device": snapshot["device"], "inode": snapshot["inode"],
                "mode": snapshot["mode"], "size": snapshot["size"],
                "mtime_ns": snapshot["mtime_ns"],
            }
            if is_json:
                assert payload is not None
                _parse_json_bytes(
                    payload, f"identity-tree JSON {relative}",
                    require_canonical=require_canonical_json)
            if hash_content:
                row["sha256"] = snapshot["sha256"]
            rows.append(row)
        else:
            raise CloseoutRecoveryError(
                f"identity tree contains non-file entry: {item}")
    for directory, before in directory_snapshots.items():
        if _directory_snapshot(directory) != before:
            raise CloseoutRecoveryError(
                f"identity directory changed during inventory: {directory}")
    return rows


def _inventory_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return _sha256_bytes(_canonical_bytes({"entries": list(rows)}))


def _protocol_lock_records(paths: CloseoutPaths) -> dict[str, Any]:
    lock = _read_json(
        paths.protocol_lock, "protocol lock", require_canonical=True)
    producers = lock.get("producer_identities")
    if lock.get("study") != STUDY or lock.get("status") != "sealed" \
            or not isinstance(producers, dict):
        raise CloseoutRecoveryError("protocol lock identity is malformed")
    required = {
        FINALIZER_RELATIVE, AUDIT_RELATIVE, FORMAL_AUDIT_RELATIVE,
        SPEC_RELATIVE,
    }
    if not required.issubset(producers):
        raise CloseoutRecoveryError("protocol lock omits closeout producers")
    verified: dict[str, Any] = {}
    for relative in sorted(required):
        record = producers[relative]
        if not isinstance(record, dict) or set(record) != {"sha256", "size"}:
            raise CloseoutRecoveryError(
                f"protocol-lock record is malformed: {relative}")
        current = _regular_file_handle(paths.repo_root / relative)
        if current["sha256"] != record["sha256"] \
                or current["size"] != record["size"]:
            raise CloseoutRecoveryError(
                f"protocol-locked producer changed: {relative}")
        verified[relative] = {
            "sha256": current["sha256"], "size": current["size"]}
    return {
        "protocol_lock": _regular_file_handle(paths.protocol_lock),
        "verified_producers": verified,
        "protocol_fingerprint": lock.get("protocol_fingerprint"),
    }


def _pre_reveal_payload(paths: CloseoutPaths, *, recorded_unix_ns: int) \
        -> dict[str, Any]:
    if (paths.final_root / "label_reveal_receipt.json").exists() \
            or paths.reveal_anchor.exists():
        raise CloseoutRecoveryError(
            "cannot create pre-reveal custody after a reveal receipt exists")
    protocol = _protocol_lock_records(paths)
    _read_json(paths.label_registry, "label custody registry",
               require_canonical=True)
    _read_json(paths.execution_registry, "execution-deck registry",
               require_canonical=True)
    _read_json(paths.raw_context_root / "summary.json",
               "raw-context producer summary", require_canonical=True)
    custody_root = paths.label_registry.parent
    custody_entries = _tree_inventory(custody_root)
    raw_entries = _tree_inventory(
        paths.raw_context_root, require_canonical_json=True)
    execution_entries = _tree_inventory(
        paths.execution_root, require_canonical_json=True)
    if not any(row.get("path") == "registry.json"
               and row.get("type") == "file" for row in execution_entries):
        raise CloseoutRecoveryError("execution tree omits mandatory registry")
    raw_summary = paths.raw_context_root / "summary.json"
    return {
        "schema": PRE_REVEAL_SCHEMA,
        "study": STUDY,
        "stage": "pre-reveal-closeout-custody",
        "status": "sealed-before-label-reveal",
        "formal_labels_read": False,
        "development_outcomes_read": False,
        "recorded_unix_ns": int(recorded_unix_ns),
        "protocol": protocol,
        "reliability_sources": {
            RECOVERY_RELATIVE: _regular_file_handle(
                paths.repo_root / RECOVERY_RELATIVE),
            WATCHDOG_RELATIVE: _regular_file_handle(
                paths.repo_root / WATCHDOG_RELATIVE),
        },
        "label_custody_registry": _regular_file_handle(paths.label_registry),
        "label_custody_tree": {
            "root": str(custody_root.resolve()),
            "entries": custody_entries,
            "entries_sha256": _inventory_sha256(custody_entries),
        },
        "raw_context_producer_summary": _regular_file_handle(raw_summary),
        "raw_context_tree": {
            "root": str(paths.raw_context_root.resolve()),
            "entries": raw_entries,
            "entries_sha256": _inventory_sha256(raw_entries),
        },
        "execution_deck_registry": _regular_file_handle(
            paths.execution_registry),
        "execution_deck_tree": {
            "root": str(paths.execution_root.resolve()),
            "entries": execution_entries,
            "entries_sha256": _inventory_sha256(execution_entries),
        },
        "paper_authorization": False,
    }


def seal_pre_reveal_custody(paths: CloseoutPaths, *,
                            now_ns: Callable[[], int] = time.time_ns) \
        -> dict[str, Any]:
    """Create or validate the mandatory execution/custody hash receipt."""

    validate_closeout_paths(paths)
    if paths.pre_reveal_receipt.exists() or paths.pre_reveal_receipt.is_symlink():
        return validate_pre_reveal_custody(paths)
    payload = _pre_reveal_payload(paths, recorded_unix_ns=now_ns())
    _atomic_json(paths.pre_reveal_receipt, payload, exclusive=True)
    return validate_pre_reveal_custody(paths, verify_raw_content=False)


def validate_pre_reveal_custody(
        paths: CloseoutPaths, *, verify_raw_content: bool = True
        ) -> dict[str, Any]:
    """Authenticate current label-free inputs against their pre-reveal receipt."""

    validate_closeout_paths(paths)
    value = _read_json(
        paths.pre_reveal_receipt, "pre-reveal custody receipt",
        require_canonical=True)
    expected_keys = {
        "schema", "study", "stage", "status", "formal_labels_read",
        "development_outcomes_read", "recorded_unix_ns", "protocol",
        "reliability_sources",
        "label_custody_registry", "label_custody_tree",
        "raw_context_producer_summary", "raw_context_tree",
        "execution_deck_registry", "execution_deck_tree",
        "paper_authorization",
    }
    if set(value) != expected_keys \
            or value.get("schema") != PRE_REVEAL_SCHEMA \
            or value.get("study") != STUDY \
            or value.get("stage") != "pre-reveal-closeout-custody" \
            or value.get("status") != "sealed-before-label-reveal" \
            or value.get("formal_labels_read") is not False \
            or value.get("development_outcomes_read") is not False \
            or value.get("paper_authorization") is not False \
            or not isinstance(value.get("recorded_unix_ns"), int) \
            or isinstance(value.get("recorded_unix_ns"), bool):
        raise CloseoutRecoveryError("pre-reveal custody receipt is malformed")
    protocol = _protocol_lock_records(paths)
    if value["protocol"] != protocol:
        raise CloseoutRecoveryError("protocol identity changed after custody seal")
    reliability = {
        RECOVERY_RELATIVE: _regular_file_handle(
            paths.repo_root / RECOVERY_RELATIVE),
        WATCHDOG_RELATIVE: _regular_file_handle(
            paths.repo_root / WATCHDOG_RELATIVE),
    }
    if value.get("reliability_sources") != reliability:
        raise CloseoutRecoveryError(
            "closeout reliability source changed after custody seal")
    expected_handles = {
        "label_custody_registry": _regular_file_handle(paths.label_registry),
        "raw_context_producer_summary": _regular_file_handle(
            paths.raw_context_root / "summary.json"),
        "execution_deck_registry": _regular_file_handle(
            paths.execution_registry),
    }
    for key, observed in expected_handles.items():
        if value.get(key) != observed:
            raise CloseoutRecoveryError(
                f"{key.replace('_', ' ')} changed after pre-reveal custody")
    custody_entries = _tree_inventory(paths.label_registry.parent)
    custody_tree = {
        "root": str(paths.label_registry.parent.resolve()),
        "entries": custody_entries,
        "entries_sha256": _inventory_sha256(custody_entries),
    }
    if value.get("label_custody_tree") != custody_tree:
        raise CloseoutRecoveryError(
            "label-custody tree changed after pre-reveal custody")
    raw_entries = _tree_inventory(
        paths.raw_context_root, hash_content=verify_raw_content,
        require_canonical_json=True)
    recorded_raw = value.get("raw_context_tree")
    if not isinstance(recorded_raw, dict):
        raise CloseoutRecoveryError("raw-context custody tree is malformed")
    if verify_raw_content:
        raw_tree = {
            "root": str(paths.raw_context_root.resolve()),
            "entries": raw_entries,
            "entries_sha256": _inventory_sha256(raw_entries),
        }
        if recorded_raw != raw_tree:
            raise CloseoutRecoveryError(
                "raw-context tree changed after pre-reveal custody")
    else:
        recorded_entries = recorded_raw.get("entries")
        if not isinstance(recorded_entries, list):
            raise CloseoutRecoveryError(
                "raw-context custody entries are malformed")
        metadata_entries = [
            {key: item for key, item in row.items() if key != "sha256"}
            for row in recorded_entries if isinstance(row, dict)]
        if len(metadata_entries) != len(recorded_entries) \
                or recorded_raw.get("root") != \
                str(paths.raw_context_root.resolve()) \
                or metadata_entries != raw_entries:
            raise CloseoutRecoveryError(
                "raw-context tree metadata changed after pre-reveal custody")
    entries = _tree_inventory(
        paths.execution_root, require_canonical_json=True)
    tree = value.get("execution_deck_tree")
    expected_tree = {
        "root": str(paths.execution_root.resolve()),
        "entries": entries,
        "entries_sha256": _inventory_sha256(entries),
    }
    if tree != expected_tree:
        raise CloseoutRecoveryError(
            "execution-deck tree changed after pre-reveal custody")
    return value


def arm_pre_reveal_interlock(paths: CloseoutPaths) -> None:
    """Prevent the ordinary launcher from entering reveal before custody check.

    The recovery state machine archives this empty directory and runs the
    unchanged finalizer in fresh staging after revalidating custody.  Existing
    post-reveal roots are never chmod'ed or otherwise mutated here.
    """

    validate_closeout_paths(paths)
    validate_pre_reveal_custody(paths, verify_raw_content=False)
    _reject_symlink_components(paths.final_root)
    receipt = paths.final_root / "label_reveal_receipt.json"
    if receipt.exists() or receipt.is_symlink():
        return
    if paths.final_root.exists() or paths.final_root.is_symlink():
        if not paths.final_root.is_dir() or paths.final_root.is_symlink():
            raise CloseoutRecoveryError("formal-finalized root is unsafe")
        if any(paths.final_root.iterdir()):
            if paths.reveal_anchor.is_file() \
                    and not paths.reveal_anchor.is_symlink():
                # A valid anchor is authenticated by recovery.  Leaving this
                # nonempty post-reveal tree untouched still makes the ordinary
                # finalizer fail its empty-output precondition.
                return
            raise CloseoutRecoveryError(
                "non-empty pre-reveal formal-finalized root is unsafe")
    else:
        paths.final_root.mkdir(mode=0o500, parents=False)
        _fsync_directory(paths.final_root.parent)
    os.chmod(paths.final_root, 0o500)
    _fsync_directory(paths.final_root)
    _fsync_directory(paths.final_root.parent)


def validate_pre_reveal_interlock(paths: CloseoutPaths) -> None:
    """Require the empty 0500 guard throughout the pre-reveal campaign."""

    validate_closeout_paths(paths)
    receipt = paths.final_root / "label_reveal_receipt.json"
    if receipt.exists() or receipt.is_symlink() \
            or paths.reveal_anchor.exists() or paths.reveal_anchor.is_symlink():
        return
    if not paths.final_root.is_dir() or paths.final_root.is_symlink() \
            or any(paths.final_root.iterdir()) \
            or stat.S_IMODE(paths.final_root.stat().st_mode) != 0o500:
        raise CloseoutRecoveryError(
            "pre-reveal formal-finalized interlock is missing or changed")


def _stop_requested(stop_sentinel: Path | None) -> bool:
    return stop_sentinel is not None and (
        stop_sentinel.exists() or stop_sentinel.is_symlink())


def _check_stop(stop_sentinel: Path | None) -> None:
    if _stop_requested(stop_sentinel):
        raise CloseoutStopRequested("closeout stop sentinel is present")


def _pipeline_processes(repo_root: Path, *, exclude_pid: int | None = None,
                        proc_root: Path = Path("/proc")) -> list[int]:
    """Return same-user closeout/campaign processes rooted in this checkout."""

    repo = Path(os.path.abspath(repo_root))
    found: list[int] = []
    try:
        entries = list(proc_root.iterdir())
    except OSError as error:
        raise CloseoutRecoveryError("cannot enumerate procfs") from error
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == exclude_pid:
            continue
        try:
            if entry.stat().st_uid != os.geteuid():
                continue
            cwd = Path(os.path.abspath(os.readlink(entry / "cwd")))
            raw = (entry / "cmdline").read_bytes()
            argv = [part.decode("utf-8", errors="strict")
                    for part in raw.split(b"\0") if part]
        except FileNotFoundError:
            continue
        except (OSError, UnicodeError) as error:
            raise CloseoutRecoveryError(
                f"cannot inspect owned process {pid}") from error
        if cwd != repo or len(argv) < 2:
            continue
        script = argv[1]
        if Path(script).is_absolute():
            try:
                script = str(Path(script).resolve().relative_to(repo))
            except (OSError, ValueError):
                continue
        if script in PIPELINE_SCRIPTS:
            found.append(pid)
    return sorted(found)


@contextmanager
def _exclusive_campaign_lock(path: Path):
    _reject_symlink_components(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(path)
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o640)
    except OSError as error:
        raise CloseoutRecoveryError("cannot open campaign lock") from error
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise CloseoutRecoveryError("campaign lock is not a regular file")
    stream = os.fdopen(descriptor, "a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise CloseoutRecoveryError("campaign lock is held") from error
        yield stream
    finally:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()


def _copy_reveal_anchor(paths: CloseoutPaths, payload: bytes) -> None:
    if paths.reveal_anchor.exists() or paths.reveal_anchor.is_symlink():
        if not paths.reveal_anchor.is_file() or paths.reveal_anchor.is_symlink() \
                or paths.reveal_anchor.read_bytes() != payload:
            raise CloseoutRecoveryError(
                "durable reveal anchor differs from partial-root receipt")
        return
    _atomic_write_bytes(paths.reveal_anchor, payload, exclusive=True)


def _receipt_candidates(paths: CloseoutPaths) -> list[Path]:
    _reject_symlink_components(paths.final_root)
    _reject_symlink_components(paths.receipt_root)
    result = [paths.reveal_anchor,
              paths.final_root / "label_reveal_receipt.json"]
    if paths.final_root.parent.is_dir():
        result.extend(sorted(paths.final_root.parent.glob(
            ".formal_finalized.recovery-staging-*/label_reveal_receipt.json")))
    return [path for path in result if path.exists() or path.is_symlink()]


def _safe_pre_reveal_partial(root: Path) -> bool:
    """Recognize only receipt-writer temporaries created before label access."""

    if not root.is_dir() or root.is_symlink():
        return False
    for item in root.iterdir():
        if not item.is_file() or item.is_symlink():
            return False
        name = item.name
        if not (
                name.startswith(".label_reveal_receipt.json.partial-")
                or (name.startswith(".label_reveal_receipt.json.")
                    and name.endswith(".tmp"))):
            return False
    return True


def _select_reveal_payload(paths: CloseoutPaths) -> bytes | None:
    payloads: list[bytes] = []
    for path in _receipt_candidates(paths):
        _reject_symlink_components(path)
        if not path.is_file() or path.is_symlink():
            raise CloseoutRecoveryError(f"reveal receipt is unsafe: {path}")
        payload = path.read_bytes()
        _parse_json_bytes(
            payload, "label-reveal receipt", require_canonical=True)
        payloads.append(payload)
    if not payloads:
        return None
    if any(payload != payloads[0] for payload in payloads[1:]):
        raise CloseoutRecoveryError("conflicting durable reveal receipts exist")
    return payloads[0]


def _archive_path(paths: CloseoutPaths, source: Path, kind: str) -> Path:
    validate_closeout_paths(paths)
    _reject_symlink_components(paths.archive_root)
    paths.archive_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    destination = paths.archive_root / (
        f"{kind}-{stamp}-{time.time_ns()}-{uuid.uuid4().hex[:12]}")
    _reject_symlink_components(destination)
    return destination


def _archive_tree(paths: CloseoutPaths, source: Path, kind: str) -> Path | None:
    _reject_symlink_components(source)
    if not source.exists() and not source.is_symlink():
        return None
    if not source.is_dir() or source.is_symlink():
        raise CloseoutRecoveryError(f"unsafe partial closeout tree: {source}")
    destination = _archive_path(paths, source, kind)
    original_mode = stat.S_IMODE(source.stat().st_mode)
    # Moving a directory across parent directories updates its ``..`` entry
    # and therefore requires owner-write permission on the directory itself.
    # The pre-reveal interlock is intentionally 0500.  Temporarily add owner
    # rwx, perform the atomic rename, then restore the evidence-tree mode.
    os.chmod(source, original_mode | stat.S_IRWXU)
    try:
        os.replace(source, destination)
    except Exception:
        os.chmod(source, original_mode)
        raise
    directory_descriptor = os.open(
        destination, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.chmod(destination, original_mode)
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    _fsync_directory(source.parent)
    _fsync_directory(destination.parent)
    return destination


def _load_finalizer(paths: CloseoutPaths) -> ModuleType:
    identities = _protocol_lock_records(paths)
    source = paths.repo_root / FINALIZER_RELATIVE
    expected_hash = identities["verified_producers"][FINALIZER_RELATIVE][
        "sha256"]
    module_name = (
        "_sage_mem_v1_sealed_finalizer_" + expected_hash[:16])
    specification = importlib.util.spec_from_file_location(module_name, source)
    if specification is None or specification.loader is None:
        raise CloseoutRecoveryError("cannot construct sealed finalizer loader")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
    except Exception as error:
        raise CloseoutRecoveryError("cannot load sealed finalizer source") \
            from error
    finally:
        sys.modules.pop(module_name, None)
    if Path(getattr(module, "__file__", "")).resolve() != source.resolve() \
            or _sha256_file(source) != expected_hash:
        raise CloseoutRecoveryError(
            "loaded finalizer identity differs from protocol lock")
    return module


def _validate_receipt_with_sealed_logic(
        finalizer: ModuleType, paths: CloseoutPaths, grid: Any,
        receipt_path: Path) -> tuple[Any, str | None]:
    raw_validator = getattr(finalizer, "_validate_raw_context_references")
    raw_references, raw_sha = raw_validator(paths.raw_context_root, grid)
    del raw_references
    finalizer._validate_reveal_receipt(  # type: ignore[attr-defined]
        receipt_path, grid, paths.label_registry,
        raw_context_sha256=raw_sha,
        execution_deck_registry=paths.execution_registry)
    return grid, raw_sha


def _write_state(paths: CloseoutPaths, status: str, **fields: Any) -> None:
    value = {
        "schema": STATE_SCHEMA, "study": STUDY,
        "stage": "closeout-recovery", "status": status,
        "updated_utc": _utc_timestamp(), "paper_authorization": False,
        **fields,
    }
    _atomic_json(paths.state_receipt, value, exclusive=False, mode=0o640)


def _write_metadata_complete_state(
        paths: CloseoutPaths, mode: str, *, hooks_injected: bool) -> None:
    if not paths.report_path.is_file() or paths.report_path.is_symlink():
        raise CloseoutRecoveryError(
            "formal audit command returned without a safe metadata report")
    _read_json(paths.report_path, "formal audit metadata report",
               require_canonical=True)
    _read_json(paths.recovery_receipt, "recovery receipt",
               require_canonical=True)
    _read_json(paths.reveal_anchor, "reveal receipt anchor",
               require_canonical=True)
    _write_state(
        paths, "metadata-audit-complete", mode=mode,
        hooks_injected=hooks_injected,
        formal_audit_report=_regular_file_handle(paths.report_path),
        recovery_receipt=_regular_file_handle(paths.recovery_receipt),
        reveal_receipt=_regular_file_handle(paths.reveal_anchor))


def _default_audit_runner(paths: CloseoutPaths) -> None:
    validate_closeout_paths(paths)
    command = [
        str(paths.repo_root / ".venv/bin/python"), AUDIT_RELATIVE,
        "--stage", "formal", "--spec", str(paths.spec_path),
        "--execute", "--resume",
    ]
    result = subprocess.run(
        command, cwd=paths.repo_root, text=True, capture_output=True,
        check=False)
    log = paths.study_root / "logs" / "formal-audit" \
        / "closeout-recovery-audit.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_bytes(
        log, (result.stdout + result.stderr).encode("utf-8"),
        exclusive=False, mode=0o640)
    if result.returncode != 0:
        raise CloseoutRecoveryError(
            f"sealed formal audit failed with return code {result.returncode}")


def _install_recovery_receipt(
        paths: CloseoutPaths, *, mode: str, receipt_payload: bytes,
        archived: Sequence[Path], summary: Mapping[str, Any],
        hooks_injected: bool) -> None:
    value = {
        "schema": RECOVERY_SCHEMA,
        "study": STUDY,
        "stage": "formal-closeout-recovery",
        "status": "validated-and-atomically-installed",
        "mode": mode,
        "recorded_utc": _utc_timestamp(),
        "phase_a_cells": summary.get("phase_a_cells"),
        "phase_a_grid_sha256": summary.get("phase_a_grid_sha256"),
        "finalized_cells": summary.get("finalized_cells"),
        "finalized_cells_sha256": summary.get("finalized_cells_sha256"),
        "label_reveal_receipt_sha256": _sha256_bytes(receipt_payload),
        "pre_reveal_custody_receipt_sha256": _sha256_file(
            paths.pre_reveal_receipt),
        "execution_deck_registry_sha256": _sha256_file(
            paths.execution_registry),
        "archives": [str(path.resolve()) for path in archived],
        "sealed_finalizer_source_sha256": _sha256_file(
            paths.repo_root / FINALIZER_RELATIVE),
        "recovery_source": _regular_file_handle(
            paths.repo_root / RECOVERY_RELATIVE),
        "watchdog_source": _regular_file_handle(
            paths.repo_root / WATCHDOG_RELATIVE),
        "hooks_injected": hooks_injected,
        "formal_audit_status": "required-after-install",
        "paper_authorization": False,
    }
    if paths.recovery_receipt.exists() or paths.recovery_receipt.is_symlink():
        existing = _read_json(
            paths.recovery_receipt, "recovery receipt",
            require_canonical=True)
        stable = {key: value for key, value in existing.items()
                  if key not in {"recorded_utc", "archives", "mode"}}
        expected = {key: item for key, item in value.items()
                    if key not in {"recorded_utc", "archives", "mode"}}
        if stable != expected:
            raise CloseoutRecoveryError(
                "existing recovery receipt conflicts with validated closeout")
        return
    _atomic_json(paths.recovery_receipt, value, exclusive=True)


def recover_closeout(
        paths: CloseoutPaths, *, stop_sentinel: Path | None = None,
        finalizer: ModuleType | None = None,
        audit_runner: Callable[[CloseoutPaths], None] = _default_audit_runner,
        process_scan: Callable[..., list[int]] = _pipeline_processes,
        current_pid: int | None = None) -> dict[str, Any]:
    """Recover or freshly restart closeout without interpreting outcomes."""

    validate_closeout_paths(paths)
    _validate_stop_sentinel(paths, stop_sentinel)
    production = (
        paths == CloseoutPaths.production(ROOT, STUDY_ROOT, SPEC_PATH))
    hooks_injected = (
        finalizer is not None or audit_runner is not _default_audit_runner
        or process_scan is not _pipeline_processes or current_pid is not None)
    if production and hooks_injected:
        raise CloseoutRecoveryError(
            "production closeout forbids injected finalizer/audit/process hooks")
    current_pid = os.getpid() if current_pid is None else int(current_pid)
    _check_stop(stop_sentinel)
    active = process_scan(paths.repo_root, exclude_pid=current_pid)
    if active:
        raise CloseoutRecoveryError(
            f"pipeline process exists before recovery lock: {active}")
    with _exclusive_campaign_lock(paths.campaign_lock):
        active = process_scan(paths.repo_root, exclude_pid=current_pid)
        if active:
            raise CloseoutRecoveryError(
                f"pipeline process exists after recovery lock: {active}")
        custody = validate_pre_reveal_custody(paths)
        _check_stop(stop_sentinel)
        module = _load_finalizer(paths) if finalizer is None else finalizer
        grid = module.validate_complete_phase_a_grid(paths.phase_a_root)
        contract = getattr(grid, "contract", None)
        if getattr(contract, "total_cells", None) != EXPECTED_CELLS:
            raise CloseoutRecoveryError("Phase-A grid is not the exact 600 grid")

        reveal_payload = _select_reveal_payload(paths)
        archived: list[Path] = []
        reveal_valid = False
        if reveal_payload is not None:
            reveal_record = _parse_json_bytes(
                reveal_payload, "label-reveal receipt",
                require_canonical=True)
            # A temporary copy permits sealed receipt validation even if the
            # only surviving copy is in interrupted staging.
            paths.receipt_root.mkdir(parents=True, exist_ok=True)
            candidate = paths.receipt_root / ".reveal-validation-candidate.json"
            _atomic_write_bytes(candidate, reveal_payload, exclusive=False)
            try:
                _validate_receipt_with_sealed_logic(
                    module, paths, grid, candidate)
            except Exception as error:
                candidate.unlink(missing_ok=True)
                raise CloseoutRecoveryError(
                    "durable reveal receipt is invalid; recovery stopped") \
                    from error
            candidate.unlink(missing_ok=True)
            reveal_time = reveal_record.get("recorded_unix_ns")
            if not isinstance(reveal_time, int) \
                    or isinstance(reveal_time, bool) \
                    or reveal_time <= custody["recorded_unix_ns"]:
                raise CloseoutRecoveryError(
                    "label-reveal receipt does not postdate custody seal")
            _copy_reveal_anchor(paths, reveal_payload)
            reveal_valid = True

        # A complete root is idempotently validated; only the missing formal
        # audit is rerun.  Invalid complete-looking roots enter reconstruction
        # only when their reveal receipt has already been authenticated.
        if reveal_valid and paths.final_root.is_dir() \
                and (paths.final_root / "summary.json").is_file():
            try:
                _read_json(
                    paths.final_root / "summary.json", "finalizer summary",
                    require_canonical=True)
                summary = module.validate_finalized_output(
                    paths.phase_a_root, paths.label_registry,
                    paths.final_root, raw_context_root=paths.raw_context_root,
                    execution_deck_registry=paths.execution_registry)
            except Exception:
                summary = None
            if summary is not None:
                _install_recovery_receipt(
                    paths, mode="already-complete", receipt_payload=reveal_payload,
                    archived=(), summary=summary,
                    hooks_injected=hooks_injected)
                _check_stop(stop_sentinel)
                audit_runner(paths)
                _write_metadata_complete_state(
                    paths, "already-complete", hooks_injected=hooks_injected)
                return {"status": "metadata-audit-complete",
                        "mode": "already-complete"}

        if paths.final_root.exists() or paths.final_root.is_symlink():
            if not paths.final_root.is_dir() or paths.final_root.is_symlink():
                raise CloseoutRecoveryError("formal-finalized root is unsafe")
            if not reveal_valid and not _safe_pre_reveal_partial(
                    paths.final_root):
                # Under the sealed finalizer, the reveal receipt is the first
                # durable file.  Non-empty content without it is not a safe
                # pre-reveal interruption and must remain untouched.
                raise CloseoutRecoveryError(
                    "non-empty partial root has no valid reveal receipt")
            archived_path = _archive_tree(
                paths, paths.final_root,
                "post-reveal-partial" if reveal_valid else "pre-reveal-empty")
            if archived_path is not None:
                archived.append(archived_path)

        stale_staging = sorted(paths.final_root.parent.glob(
            ".formal_finalized.recovery-staging-*"))
        for stale in stale_staging:
            if not reveal_valid and not _safe_pre_reveal_partial(stale):
                raise CloseoutRecoveryError(
                    "non-empty interrupted staging has no valid reveal receipt")
            archived_path = _archive_tree(paths, stale, "interrupted-staging")
            if archived_path is not None:
                archived.append(archived_path)

        _check_stop(stop_sentinel)
        staging = paths.final_root.parent / (
            f".formal_finalized.recovery-staging-{uuid.uuid4().hex}")
        _reject_symlink_components(staging)
        if staging.exists() or staging.is_symlink():
            raise CloseoutRecoveryError("fresh recovery staging already exists")

        mode = "post-reveal-reconstruction" if reveal_valid \
            else "pre-reveal-fresh-restart"
        _write_state(paths, "reconstructing", mode=mode,
                     archived=[str(path.resolve()) for path in archived],
                     hooks_injected=hooks_injected)

        original_writer = getattr(module, "_record_label_reveal_receipt", None)
        if original_writer is None:
            raise CloseoutRecoveryError(
                "sealed finalizer lacks receipt writer boundary")

        def guarded_receipt_writer(
                validated_grid: Any, registry_path: Path,
                output_root: Path, *, raw_context_sha256: str | None = None,
                execution_deck_registry: Path | None = None) -> Path:
            # Reauthenticate every custody tree at the finalizer's exact
            # reveal boundary.  This callback completes before the unchanged
            # caller opens any semantic labels.
            if Path(registry_path).resolve() != paths.label_registry.resolve() \
                    or execution_deck_registry is None \
                    or Path(execution_deck_registry).resolve() != \
                    paths.execution_registry.resolve():
                raise CloseoutRecoveryError(
                    "sealed finalizer requested different closeout inputs")
            validate_pre_reveal_custody(paths)
            current_grid = module.validate_complete_phase_a_grid(
                paths.phase_a_root)
            if getattr(current_grid, "grid_sha256", None) != \
                    getattr(validated_grid, "grid_sha256", None):
                raise CloseoutRecoveryError(
                    "Phase-A grid changed at the reveal boundary")
            if raw_context_sha256 is None:
                raise CloseoutRecoveryError(
                    "sealed finalizer omitted raw-context identity")
            if reveal_valid:
                _validate_receipt_with_sealed_logic(
                    module, paths, current_grid, paths.reveal_anchor)
                destination = Path(output_root) / "label_reveal_receipt.json"
                _atomic_write_bytes(
                    destination, reveal_payload, exclusive=True, mode=0o440)
                if destination.read_bytes() != reveal_payload:
                    raise CloseoutRecoveryError(
                        "recovered reveal receipt bytes changed")
                return destination
            destination = original_writer(
                validated_grid, registry_path, output_root,
                raw_context_sha256=raw_context_sha256,
                execution_deck_registry=execution_deck_registry)
            _read_json(
                destination, "fresh label-reveal receipt",
                require_canonical=True)
            _validate_receipt_with_sealed_logic(
                module, paths, current_grid, destination)
            return destination

        setattr(module, "_record_label_reveal_receipt",
                guarded_receipt_writer)
        try:
            summary = module.finalize_formal_grid(
                paths.phase_a_root, paths.label_registry, staging,
                raw_context_root=paths.raw_context_root,
                execution_deck_registry=paths.execution_registry)
        finally:
            setattr(module, "_record_label_reveal_receipt", original_writer)

        _check_stop(stop_sentinel)
        _read_json(
            staging / "summary.json", "reconstructed finalizer summary",
            require_canonical=True)
        validated = module.validate_finalized_output(
            paths.phase_a_root, paths.label_registry, staging,
            raw_context_root=paths.raw_context_root,
            execution_deck_registry=paths.execution_registry)
        if validated != summary:
            raise CloseoutRecoveryError(
                "sealed finalizer validation summary differs after reconstruction")
        _fsync_directory_tree(staging)
        staged_receipt = staging / "label_reveal_receipt.json"
        if not staged_receipt.is_file() or staged_receipt.is_symlink():
            raise CloseoutRecoveryError("staging lacks a safe reveal receipt")
        staged_payload = staged_receipt.read_bytes()
        if reveal_valid and staged_payload != reveal_payload:
            raise CloseoutRecoveryError(
                "staging did not preserve original reveal receipt bytes")
        if not reveal_valid:
            # This receipt was durably created by the unchanged finalizer before
            # it opened labels.  Preserve it as the recovery anchor now.
            _copy_reveal_anchor(paths, staged_payload)
            reveal_payload = staged_payload
            _validate_receipt_with_sealed_logic(
                module, paths, grid, paths.reveal_anchor)

        _check_stop(stop_sentinel)
        if paths.final_root.exists() or paths.final_root.is_symlink():
            raise CloseoutRecoveryError(
                "formal-finalized destination reappeared before install")
        os.replace(staging, paths.final_root)
        _fsync_directory(paths.final_root.parent)
        _read_json(
            paths.final_root / "summary.json", "installed finalizer summary",
            require_canonical=True)
        installed = module.validate_finalized_output(
            paths.phase_a_root, paths.label_registry, paths.final_root,
            raw_context_root=paths.raw_context_root,
            execution_deck_registry=paths.execution_registry)
        if installed != summary:
            raise CloseoutRecoveryError(
                "atomically installed output differs from validated staging")
        _install_recovery_receipt(
            paths, mode=mode, receipt_payload=reveal_payload,
            archived=archived, summary=summary,
            hooks_injected=hooks_injected)
        _check_stop(stop_sentinel)
        audit_runner(paths)
        _write_metadata_complete_state(
            paths, mode, hooks_injected=hooks_injected)
        return {"status": "metadata-audit-complete", "mode": mode}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--seal-and-arm", action="store_true")
    action.add_argument("--recover", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--study-root", type=Path, default=STUDY_ROOT)
    parser.add_argument("--spec", type=Path, default=SPEC_PATH)
    parser.add_argument("--stop-sentinel", type=Path)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def recovery_argv(paths: CloseoutPaths, stop_sentinel: Path) -> tuple[str, ...]:
    return (
        str(paths.repo_root / ".venv/bin/python"),
        "scripts/recover_sage_mem_v1_closeout.py",
        "--recover", "--repo-root", str(paths.repo_root),
        "--study-root", str(paths.study_root),
        "--spec", str(paths.spec_path),
        "--stop-sentinel", str(Path(os.path.abspath(stop_sentinel))),
        "--execute",
    )


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    paths = CloseoutPaths.production(
        args.repo_root, args.study_root, args.spec)
    if not args.execute:
        print(json.dumps({
            "study": STUDY,
            "preview": True,
            "action": "seal-and-arm" if args.seal_and_arm else "recover",
            "outcomes_interpreted": False,
            "paper_authorization": False,
        }, sort_keys=True))
        return 0
    if args.seal_and_arm:
        seal_pre_reveal_custody(paths)
        arm_pre_reveal_interlock(paths)
        print(json.dumps({
            "study": STUDY, "status": "pre-reveal-closeout-armed",
            "paper_authorization": False,
        }, sort_keys=True))
        return 0
    try:
        result = recover_closeout(
            paths, stop_sentinel=args.stop_sentinel)
    except CloseoutStopRequested:
        _write_state(paths, "stopped", paper_authorization=False)
        return 75
    print(json.dumps({**result, "study": STUDY,
                      "paper_authorization": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CloseoutRecoveryError as error:
        print(f"SAGE-Mem closeout stopped: {error}", file=sys.stderr)
        raise SystemExit(2) from error
