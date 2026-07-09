#!/usr/bin/env python3
"""Identity-bound, fail-closed watchdog for the SAGE-Mem v1 campaign.

This utility is intentionally separate from the sealed experiment runner.  It
does not import experiment code, open measurement artifacts, or interpret any
experimental outcome.  It only observes process identity, resource headroom,
artifact *completion metadata*, and the existence of registered cell
manifests.  A metadata-complete report only stops this operational monitor; it
is never an authorization to update the paper or make a scientific claim.

The initial supervisor PID and tmux session are supplied explicitly.  Every
automatic restart is launched in a new tmux session and then rebound to an
exact ``/proc`` identity (PID, start time, cwd, and argv).  Resource failure
therefore terminates the currently tracked session, including a session that
the watchdog created after a restart.

Creating the stop-sentinel asks the watchdog to exit without relaunching.  It
does not terminate a healthy campaign; an operator can create the sentinel
before intentionally stopping the campaign session.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import stat
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence


ROOT = Path(__file__).resolve().parents[1]
STUDY_ROOT = ROOT / "outputs/sage_mem_v1"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORT_SCHEMA = "sage_mem_v1_formal_evidence_audit_v1"
REPORT_STUDY = "sage-mem-v1"
REPORT_STAGE = "formal-evidence-audit"
REPORT_STATUS = "complete"
FORMAL_CELL_COUNT = 600
FINALIZER_SCHEMA = "sage_mem_v1_formal_finalizer_v1"
PHASE_A_SCHEMA = "sage_mem_v1_phase_a_cell_v1"
RECOVERY_STATE_SCHEMA = "sage_mem_v1_closeout_state_v1"

REPORT_KEYS = {
    "schema", "study", "stage", "status", "phase_a_cells_verified",
    "finalized_cells_verified", "phase_a_grid_sha256",
    "identity_ledger_sha256", "comparators_verified", "resources_verified",
    "raw_context_references_verified", "bootstrap_draws_per_contrast",
    "cohorts", "execution_program", "prior_can_substitute_for_host_output",
    "per_age_claims_only", "pooled_cross_host_score_computed",
    "universal_success_claim_permitted",
}
FINALIZER_SUMMARY_KEYS = {
    "schema", "study", "stage", "status", "phase_a_cells",
    "phase_a_grid_sha256", "label_reveal_receipt_sha256",
    "label_registry_sha256", "development_outcomes_read",
    "per_age_results_preserved", "pointmaze_x4_native_clustering_preserved",
    "raw_context_reference", "execution_decks", "finalized_cells_sha256",
    "finalized_cells",
}

FORMAL_CONFIRMATION = "RUN_SAGE_MEM_V1_FORMAL"
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

GIB = 1024 ** 3
DEFAULT_DISK_WARN_BYTES = 250 * GIB
DEFAULT_DISK_STOP_BYTES = 200 * GIB
DEFAULT_RAM_STOP_BYTES = 64 * GIB


class WatchdogError(RuntimeError):
    """The watchdog cannot continue without weakening an invariant."""


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _has_symlink_component(path: Path) -> bool:
    lexical = Path(os.path.abspath(path))
    for component in [*reversed(lexical.parents), lexical]:
        try:
            if component.is_symlink():
                return True
        except OSError:
            return True
    return False


class EventLog:
    """Append one fsync-backed JSON object per watchdog event."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def emit(self, event: str, **fields: Any) -> None:
        record = {"timestamp_utc": _utc_timestamp(), "event": event, **fields}
        if _has_symlink_component(self.path):
            raise WatchdogError("watchdog event-log path contains a symlink")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if _has_symlink_component(self.path):
            raise WatchdogError("watchdog event-log path contains a symlink")
        existed = self.path.exists() or self.path.is_symlink()
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(
            self.path, flags, 0o640)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise WatchdogError("watchdog event log is not a regular file")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            payload = (_canonical_json(record) + "\n").encode("utf-8")
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise WatchdogError("cannot append watchdog event")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(descriptor)
        if not existed:
            parent = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(parent)
            finally:
                os.close(parent)


@dataclass(frozen=True)
class CompletionMetadataCheck:
    """Outcome-blind operational metadata status, never paper authorization."""

    state: str
    reason: str | None = None
    phase_a_grid_sha256: str | None = None


def _read_metadata_json(path: Path, label: str) -> tuple[dict[str, Any] | None,
                                                        str | None]:
    """Read one regular, non-symlink JSON mapping without outcome semantics."""

    path = Path(path)
    if not path.exists() and not path.is_symlink():
        return None, f"{label}-absent"
    if not path.is_file() or path.is_symlink() \
            or _has_symlink_component(path):
        return None, f"{label}-is-not-a-regular-file"
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = item
        return result
    try:
        payload = path.read_bytes()
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token: {token}")))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return None, f"{label}-is-not-valid-json"
    if not isinstance(value, dict):
        return None, f"{label}-is-not-a-mapping"
    expected = (_canonical_json(value) + "\n").encode("utf-8")
    if payload != expected:
        return None, f"{label}-is-not-canonical-json"
    return value, None


def _expected_manifest_paths(root: Path) -> set[Path]:
    return {
        root / cohort / arm / f"seed-{seed}" / "manifest.json"
        for cohort in COHORTS for arm in ARMS for seed in SEEDS
    }


def _validate_exact_cell_inventory(root: Path, label: str) -> str | None:
    """Require exactly the registered 600 regular manifests and no extras."""

    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        return f"{label}-root-is-not-a-regular-directory"
    expected = _expected_manifest_paths(root)
    actual = set(root.rglob("manifest.json"))
    if actual != expected:
        return f"{label}-manifest-inventory-mismatch"
    if any(not path.is_file() or path.is_symlink() for path in expected):
        return f"{label}-manifest-is-not-a-regular-file"
    if any(root.rglob(".seed-*.partial-*")):
        return f"{label}-partial-cell-present"
    return None


def _sha256_file(path: Path) -> str:
    if _has_symlink_component(path):
        raise WatchdogError(f"identity path contains symlink: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_cross_bound_cells(study_root: Path, grid_hash: str) -> str | None:
    """Bind every finalized identity to its current Phase-A manifest."""

    phase_root = study_root / "cells"
    finalized_root = study_root / "formal_finalized" / "cells"
    for cohort in COHORTS:
        for arm in ARMS:
            for seed in SEEDS:
                phase_path = (phase_root / cohort / arm / f"seed-{seed}"
                              / "manifest.json")
                final_path = (finalized_root / cohort / arm / f"seed-{seed}"
                              / "manifest.json")
                phase, error = _read_metadata_json(
                    phase_path, "phase-a-manifest")
                if error is not None or phase is None:
                    return error
                phase_expected = {
                    "schema": PHASE_A_SCHEMA,
                    "study": REPORT_STUDY,
                    "stage": "formal-phase-a",
                    "status": "complete-label-free",
                    "cohort": cohort,
                    "arm": arm,
                    "seed": seed,
                }
                if any(phase.get(key) != wanted
                       for key, wanted in phase_expected.items()):
                    return "phase-a-manifest-identity-mismatch"
                finalized, error = _read_metadata_json(
                    final_path, "finalized-manifest")
                if error is not None or finalized is None:
                    return error
                finalized_expected = {
                    "schema": FINALIZER_SCHEMA,
                    "study": REPORT_STUDY,
                    "stage": "formal-finalized",
                    "status": REPORT_STATUS,
                    "cohort": cohort,
                    "arm": arm,
                    "seed": seed,
                    "phase_a_grid_sha256": grid_hash,
                    "phase_a_manifest_sha256": _sha256_file(phase_path),
                }
                if any(finalized.get(key) != wanted
                       for key, wanted in finalized_expected.items()):
                    return "finalized-manifest-cross-binding-mismatch"
    return None


def validate_completion_metadata(
        path: Path, study_root: Path) -> CompletionMetadataCheck:
    """Cross-bind only completion metadata to the current finalized grid.

    This deliberately does not interpret effects or gates and cannot authorize
    manuscript text.  Publication requires the independent result and paper
    binding audits outside this watchdog.
    """

    path = Path(path)
    if not path.exists() and not path.is_symlink():
        return CompletionMetadataCheck("absent")
    value, error = _read_metadata_json(path, "report")
    if error is not None or value is None:
        return CompletionMetadataCheck("invalid", error)
    if set(value) != REPORT_KEYS:
        return CompletionMetadataCheck(
            "invalid", "report-top-level-keys-mismatch")
    expected = {
        "schema": REPORT_SCHEMA,
        "study": REPORT_STUDY,
        "stage": REPORT_STAGE,
        "status": REPORT_STATUS,
        "phase_a_cells_verified": FORMAL_CELL_COUNT,
        "finalized_cells_verified": FORMAL_CELL_COUNT,
    }
    for key, wanted in expected.items():
        observed = value.get(key)
        if observed != wanted or (key.endswith("cells_verified")
                                  and isinstance(observed, bool)):
            return CompletionMetadataCheck(
                "invalid", f"report-{key}-mismatch")
    grid_hash = value.get("phase_a_grid_sha256")
    identity_hash = value.get("identity_ledger_sha256")
    if not _is_sha256(grid_hash) or not _is_sha256(identity_hash):
        return CompletionMetadataCheck(
            "invalid", "report-identity-hash-malformed")

    root = Path(study_root)
    inventory_error = _validate_exact_cell_inventory(
        root / "cells", "phase-a")
    if inventory_error is not None:
        return CompletionMetadataCheck("invalid", inventory_error)
    inventory_error = _validate_exact_cell_inventory(
        root / "formal_finalized" / "cells", "finalized")
    if inventory_error is not None:
        return CompletionMetadataCheck("invalid", inventory_error)

    summary, error = _read_metadata_json(
        root / "formal_finalized" / "summary.json", "finalizer-summary")
    if error is not None or summary is None:
        return CompletionMetadataCheck("invalid", error)
    if set(summary) != FINALIZER_SUMMARY_KEYS:
        return CompletionMetadataCheck(
            "invalid", "finalizer-summary-top-level-keys-mismatch")
    summary_expected = {
        "schema": FINALIZER_SCHEMA,
        "study": REPORT_STUDY,
        "stage": "formal-finalizer",
        "status": REPORT_STATUS,
        "phase_a_cells": FORMAL_CELL_COUNT,
        "finalized_cells": FORMAL_CELL_COUNT,
    }
    for key, wanted in summary_expected.items():
        observed = summary.get(key)
        if observed != wanted or (key.endswith("cells")
                                  and isinstance(observed, bool)):
            return CompletionMetadataCheck(
                "invalid", f"finalizer-summary-{key}-mismatch")
    summary_grid_hash = summary.get("phase_a_grid_sha256")
    if not _is_sha256(summary_grid_hash) \
            or summary_grid_hash != grid_hash:
        return CompletionMetadataCheck(
            "invalid", "report-finalizer-grid-hash-mismatch")
    for key in ("label_reveal_receipt_sha256", "label_registry_sha256",
                "finalized_cells_sha256"):
        if not _is_sha256(summary.get(key)):
            return CompletionMetadataCheck(
                "invalid", f"finalizer-summary-{key}-malformed")
    cross_binding_error = _validate_cross_bound_cells(
        root, str(summary_grid_hash))
    if cross_binding_error is not None:
        return CompletionMetadataCheck("invalid", cross_binding_error)
    return CompletionMetadataCheck(
        "complete", phase_a_grid_sha256=str(grid_hash))


def validate_recovery_metadata_state(
        study_root: Path, report_path: Path) -> CompletionMetadataCheck:
    """Authenticate the recovery wrapper's metadata-only terminal state."""

    root = Path(study_root)
    path = root / "receipts" / "closeout" / "recovery_state.json"
    value, error = _read_metadata_json(path, "closeout-recovery-state")
    if error == "closeout-recovery-state-absent":
        return CompletionMetadataCheck("absent")
    if error is not None or value is None:
        return CompletionMetadataCheck("invalid", error)
    expected_keys = {
        "schema", "study", "stage", "status", "updated_utc",
        "paper_authorization", "hooks_injected", "mode", "formal_audit_report",
        "recovery_receipt", "reveal_receipt",
    }
    if set(value) != expected_keys \
            or value.get("schema") != RECOVERY_STATE_SCHEMA \
            or value.get("study") != REPORT_STUDY \
            or value.get("stage") != "closeout-recovery" \
            or value.get("status") != "metadata-audit-complete" \
            or value.get("paper_authorization") is not False \
            or value.get("hooks_injected") is not False \
            or not isinstance(value.get("updated_utc"), str) \
            or not value["updated_utc"] \
            or not isinstance(value.get("mode"), str) or not value["mode"]:
        return CompletionMetadataCheck(
            "invalid", "closeout-recovery-state-identity-mismatch")
    expected_paths = {
        "formal_audit_report": Path(report_path),
        "recovery_receipt": (
            root / "receipts/closeout/recovery_receipt.json"),
        "reveal_receipt": (
            root / "receipts/closeout/label_reveal_receipt.original.json"),
    }
    for key, expected_path in expected_paths.items():
        record = value.get(key)
        if not isinstance(record, dict) or set(record) != {
                "path", "sha256", "size"}:
            return CompletionMetadataCheck(
                "invalid", f"closeout-recovery-{key}-handle-malformed")
        expected = Path(os.path.abspath(expected_path))
        if not expected.is_file() or expected.is_symlink() \
                or record.get("path") != str(expected.resolve()) \
                or record.get("sha256") != _sha256_file(expected) \
                or record.get("size") != expected.stat().st_size:
            return CompletionMetadataCheck(
                "invalid", f"closeout-recovery-{key}-handle-mismatch")
    return CompletionMetadataCheck("complete")


def _absolute_lexical(path: str, cwd: Path) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = cwd / value
    return Path(os.path.abspath(value))


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    uid: int
    ppid: int
    pgrp: int
    session: int
    start_ticks: int
    state: str
    cwd: Path
    argv: tuple[str, ...]

    @property
    def live(self) -> bool:
        return self.state != "Z"


class ProcessInspector:
    """Read exact same-user process identities from procfs."""

    def __init__(self, proc_root: Path = Path("/proc"),
                 expected_uid: int | None = None):
        self.proc_root = Path(proc_root)
        self.expected_uid = (os.geteuid() if expected_uid is None
                             else int(expected_uid))

    def read(self, pid: int) -> ProcessIdentity | None:
        root = self.proc_root / str(int(pid))
        try:
            raw_stat = (root / "stat").read_text(encoding="utf-8")
            close = raw_stat.rfind(")")
            if close < 0:
                raise ValueError("malformed proc stat")
            tail = raw_stat[close + 2:].split()
            state = tail[0]
            ppid = int(tail[1])
            pgrp = int(tail[2])
            session = int(tail[3])
            start_ticks = int(tail[19])
            uid = int(root.stat().st_uid)
            raw_argv = (root / "cmdline").read_bytes()
            argv = tuple(
                part.decode("utf-8", errors="strict")
                for part in raw_argv.split(b"\0") if part)
            cwd = Path(os.readlink(root / "cwd"))
        except FileNotFoundError:
            return None
        except (OSError, UnicodeError, ValueError, IndexError) as error:
            raise WatchdogError(f"cannot inspect process {pid}") from error
        if not argv:
            return None
        return ProcessIdentity(
            pid=int(pid), uid=uid, ppid=ppid, pgrp=pgrp, session=session,
            start_ticks=start_ticks, state=state,
            cwd=Path(os.path.abspath(cwd)), argv=argv)

    def scan(self) -> list[ProcessIdentity]:
        result: list[ProcessIdentity] = []
        try:
            entries = list(self.proc_root.iterdir())
        except OSError as error:
            raise WatchdogError("cannot enumerate procfs") from error
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                identity = self.read(int(entry.name))
            except WatchdogError:
                # Unrelated nondumpable processes may hide cwd/cmdline.  The
                # session-specific verifier below handles owned members more
                # strictly and must not silently omit them.
                continue
            if identity is not None and identity.live \
                    and identity.uid == self.expected_uid:
                result.append(identity)
        return result

    def same_process(self, expected: ProcessIdentity) -> bool:
        observed = self.read(expected.pid)
        return observed is not None and observed.live \
            and observed.uid == self.expected_uid \
            and observed.uid == expected.uid \
            and observed.ppid == expected.ppid \
            and observed.pgrp == expected.pgrp \
            and observed.session == expected.session \
            and observed.start_ticks == expected.start_ticks \
            and observed.cwd == expected.cwd \
            and observed.argv == expected.argv

    def session_members(self, expected: ProcessIdentity) \
            -> list[ProcessIdentity]:
        """Return all same-owner members of the tracked kernel session."""

        if expected.uid != self.expected_uid or expected.session <= 0:
            return []
        try:
            entries = list(self.proc_root.iterdir())
        except OSError as error:
            raise WatchdogError("cannot enumerate procfs") from error
        result: list[ProcessIdentity] = []
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                if entry.stat().st_uid != expected.uid:
                    continue
                raw_stat = (entry / "stat").read_text(encoding="utf-8")
                close = raw_stat.rfind(")")
                if close < 0:
                    raise ValueError("malformed proc stat")
                tail = raw_stat[close + 2:].split()
                state = tail[0]
                session = int(tail[3])
            except FileNotFoundError:
                continue
            except (OSError, UnicodeError, ValueError, IndexError) as error:
                raise WatchdogError(
                    f"cannot inspect owned process {entry.name}") from error
            if state == "Z" or session != expected.session:
                continue
            identity = self.read(int(entry.name))
            if identity is None:
                # A process can disappear between the lightweight session read
                # and full identity read.  Recheck before treating it as gone.
                try:
                    if not entry.exists():
                        continue
                except OSError as error:
                    raise WatchdogError(
                        f"cannot recheck owned process {entry.name}") from error
                raise WatchdogError(
                    f"live tracked-session process {entry.name} lacks identity")
            if identity.uid != expected.uid \
                    or identity.session != expected.session:
                raise WatchdogError("tracked-session identity changed during scan")
            result.append(identity)
        return result

    def signal_session(self, expected: ProcessIdentity, signum: int) -> bool:
        """Signal every verified process group in the tracked kernel session."""

        members = self.session_members(expected)
        if not members:
            return True
        groups: dict[int, list[ProcessIdentity]] = {}
        for process in members:
            groups.setdefault(process.pgrp, []).append(process)
        success = True
        for pgrp, candidates in groups.items():
            verified = False
            for candidate in candidates:
                observed = self.read(candidate.pid)
                if observed is None or not observed.live \
                        or observed.uid != expected.uid \
                        or observed.start_ticks != candidate.start_ticks \
                        or observed.session != expected.session \
                        or observed.pgrp != pgrp:
                    continue
                try:
                    if os.getsid(candidate.pid) != expected.session \
                            or os.getpgid(candidate.pid) != pgrp:
                        continue
                    os.killpg(pgrp, signum)
                    verified = True
                    break
                except ProcessLookupError:
                    verified = True
                    break
                except OSError:
                    success = False
                    verified = True
                    break
            if not verified:
                success = False
        return success


def _expected_python(repo_root: Path) -> Path:
    return Path(os.path.abspath(repo_root / ".venv/bin/python"))


def supervisor_argv(repo_root: Path, poll_seconds: float = 30.0) \
        -> tuple[str, ...]:
    poll = str(int(poll_seconds)) if float(poll_seconds).is_integer() \
        else str(float(poll_seconds))
    return (
        str(_expected_python(repo_root)),
        "scripts/run_sage_mem_v1_campaign.py",
        "--poll-seconds", poll,
        "--execute", "--resume",
    )


def is_exact_supervisor(
        process: ProcessIdentity, repo_root: Path,
        poll_seconds: float = 30.0) -> bool:
    repo = Path(os.path.abspath(repo_root))
    if process.uid != os.geteuid() or process.cwd != repo \
            or process.session != process.pid or process.pgrp != process.pid \
            or len(process.argv) < 1:
        return False
    normalized = (
        str(_absolute_lexical(process.argv[0], process.cwd)),
        *process.argv[1:],
    )
    return tuple(normalized) == supervisor_argv(repo, poll_seconds)


def is_exact_full_launcher(process: ProcessIdentity, repo_root: Path) -> bool:
    repo = Path(os.path.abspath(repo_root))
    if process.uid != os.geteuid() or process.cwd != repo or not process.argv:
        return False
    normalized = (
        str(_absolute_lexical(process.argv[0], process.cwd)),
        *process.argv[1:],
    )
    expected = (
        str(_expected_python(repo)),
        "scripts/launch_sage_mem_v1.py",
        "--spec", str(repo / "configs/sage_mem_v1.yaml"),
        "--execute", "--stage", "full", "--resume",
        "--formal-confirmation", FORMAL_CONFIRMATION,
    )
    return tuple(normalized) == expected


def is_exact_full_worker(process: ProcessIdentity, repo_root: Path) -> bool:
    """Recognize one exact registered Phase-A worker command."""

    repo = Path(os.path.abspath(repo_root))
    if process.uid != os.geteuid() or process.cwd != repo or not process.argv:
        return False
    normalized = (
        str(_absolute_lexical(process.argv[0], process.cwd)),
        *process.argv[1:],
    )
    return (
        len(normalized) == 16
        and normalized[0] == str(_expected_python(repo))
        and normalized[1:7] == (
            "scripts/run_sage_mem_v1.py", "--stage", "full", "--spec",
            str(repo / "configs/sage_mem_v1.yaml"), "--execute")
        and normalized[7] == "--cohort" and normalized[8] in COHORTS
        and normalized[9] == "--arm" and normalized[10] in ARMS
        and normalized[11] == "--seed" and normalized[12].isdigit()
        and int(normalized[12]) in SEEDS
        and normalized[13] == "--formal-confirmation"
        and normalized[14] == FORMAL_CONFIRMATION
        and normalized[15] == "--resume"
    )


def is_full_pipeline_process(process: ProcessIdentity,
                             repo_root: Path) -> bool:
    """Recognize exact launchers/workers, including their natural argv."""

    if is_exact_full_launcher(process, repo_root):
        return True
    if is_exact_full_worker(process, repo_root):
        return True
    repo = Path(os.path.abspath(repo_root))
    if process.uid != os.geteuid() or process.cwd != repo \
            or len(process.argv) < 2:
        return False
    script = _absolute_lexical(process.argv[1], process.cwd)
    return script in {
        repo / "scripts/run_sage_mem_v1_campaign.py",
        repo / "scripts/launch_sage_mem_v1.py",
        repo / "scripts/run_sage_mem_v1.py",
        repo / "scripts/sage_mem_v1_formal_finalizer.py",
        repo / "scripts/audit_sage_mem_v1.py",
        repo / "scripts/audit_sage_mem_v1_formal.py",
        repo / "scripts/recover_sage_mem_v1_closeout.py",
    }


def closeout_recovery_argv(repo_root: Path, study_root: Path,
                           stop_sentinel: Path) -> tuple[str, ...]:
    """Return the sole production argv accepted for closeout recovery."""

    repo = Path(os.path.abspath(repo_root))
    study = Path(os.path.abspath(study_root))
    stop = Path(os.path.abspath(stop_sentinel))
    return (
        str(_expected_python(repo)),
        "scripts/recover_sage_mem_v1_closeout.py",
        "--recover", "--repo-root", str(repo),
        "--study-root", str(study),
        "--spec", str(repo / "configs/sage_mem_v1.yaml"),
        "--stop-sentinel", str(stop),
        "--execute",
    )


def is_exact_closeout_recovery(
        process: ProcessIdentity, repo_root: Path, study_root: Path,
        stop_sentinel: Path) -> bool:
    """Recognize one exact same-user closeout recovery kernel process."""

    repo = Path(os.path.abspath(repo_root))
    if process.uid != os.geteuid() or process.cwd != repo or not process.argv:
        return False
    normalized = (
        str(_absolute_lexical(process.argv[0], process.cwd)),
        *process.argv[1:],
    )
    return tuple(normalized) == closeout_recovery_argv(
        repo, study_root, stop_sentinel)


@dataclass(frozen=True)
class PaneIdentity:
    pid: int
    dead: bool
    dead_status: int | None


class TmuxClient:
    def __init__(self, executable: str = "tmux"):
        self.executable = executable

    def _run(self, args: Sequence[str]) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                [self.executable, *args], text=True, capture_output=True,
                check=False)
        except OSError as error:
            raise WatchdogError("cannot execute tmux") from error

    def pane(self, session: str) -> PaneIdentity | None:
        shown = self._run([
            "display-message", "-p", "-t", session, "#{session_name}"])
        if shown.returncode != 0 or shown.stdout.strip() != session:
            return None
        result = self._run([
            "list-panes", "-t", session, "-F",
            "#{pane_pid}\t#{pane_dead}\t#{pane_dead_status}"])
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if result.returncode != 0 or len(lines) != 1:
            return None
        fields = lines[0].split("\t")
        if len(fields) != 3 or not fields[0].isdigit() \
                or fields[1] not in {"0", "1"}:
            return None
        status = int(fields[2]) if fields[2].lstrip("-").isdigit() else None
        return PaneIdentity(
            pid=int(fields[0]), dead=fields[1] == "1", dead_status=status)

    def kill(self, session: str) -> bool:
        result = self._run(["kill-session", "-t", session])
        return result.returncode == 0

    def launch(self, session: str, repo_root: Path, command: Sequence[str],
               log_path: Path) -> bool:
        if _has_symlink_component(log_path):
            raise WatchdogError("tmux log path contains a symlink")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if _has_symlink_component(log_path) \
                or (log_path.exists() and not log_path.is_file()):
            raise WatchdogError("tmux log path is unsafe")
        shell = (
            "exec env PYTHONUNBUFFERED=1 " + shlex.join(command)
            + " >>" + shlex.quote(str(log_path)) + " 2>&1")
        created = self._run([
            "new-session", "-d", "-s", session, "-c", str(repo_root),
            "bash", "-lc", shell])
        if created.returncode != 0:
            return False
        option = self._run([
            "set-option", "-t", session, "remain-on-exit", "on"])
        if option.returncode != 0:
            self.kill(session)
            return False
        return True


def preflight_bootstrap_identity(
        *, repo_root: Path, supervisor_pid: int, tmux_session: str,
        campaign_poll_seconds: float, inspector: ProcessInspector,
        tmux: TmuxClient) -> ProcessIdentity:
    """Authenticate the supplied supervisor/session before any state mutation."""

    repo = Path(os.path.abspath(repo_root))
    process = inspector.read(supervisor_pid)
    if process is None or not process.live \
            or not is_exact_supervisor(
                process, repo, campaign_poll_seconds):
        raise WatchdogError("bootstrap supervisor identity differs")
    pane = tmux.pane(tmux_session)
    if pane is None or pane.dead or pane.pid != supervisor_pid:
        raise WatchdogError("bootstrap tmux session does not own supervisor")
    return process


@dataclass(frozen=True)
class ResourceSnapshot:
    disk_available: int
    memory_available: int


def read_resources(path: Path) -> ResourceSnapshot:
    try:
        filesystem = os.statvfs(path)
        disk = int(filesystem.f_bavail * filesystem.f_frsize)
        memory: int | None = None
        with Path("/proc/meminfo").open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("MemAvailable:"):
                    fields = line.split()
                    memory = int(fields[1]) * 1024
                    break
    except (OSError, ValueError, IndexError) as error:
        raise WatchdogError("cannot read host resource headroom") from error
    if memory is None:
        raise WatchdogError("MemAvailable is absent from /proc/meminfo")
    return ResourceSnapshot(disk_available=disk, memory_available=memory)


def read_gpu_temperatures(
        physical_gpus: Sequence[int] = (0, 1, 2),
        executable: str = "nvidia-smi") -> dict[int, int]:
    """Read owned-GPU temperatures without mutating any GPU setting."""

    result = subprocess.run([
        executable, "--query-gpu=index,temperature.gpu",
        "--format=csv,noheader,nounits",
    ], text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise WatchdogError("nvidia-smi temperature query failed")
    observed: dict[int, int] = {}
    try:
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            index_text, temperature_text = (
                field.strip() for field in line.split(",", maxsplit=1))
            observed[int(index_text)] = int(temperature_text)
    except (TypeError, ValueError) as error:
        raise WatchdogError("nvidia-smi temperature output is malformed") \
            from error
    requested = tuple(map(int, physical_gpus))
    if any(index not in observed for index in requested):
        raise WatchdogError("one or more owned GPU temperatures are absent")
    return {index: observed[index] for index in requested}


def count_formal_cell_manifests(study_root: Path) -> int:
    root = Path(study_root) / "cells"
    count = 0
    for cohort in COHORTS:
        for arm in ARMS:
            for seed in SEEDS:
                path = root / cohort / arm / f"seed-{seed}" / "manifest.json"
                if path.is_file() and not path.is_symlink():
                    count += 1
    return count


def progress_token(study_root: Path) -> tuple[int, int]:
    finalized = Path(study_root) / "formal_finalized" / "summary.json"
    return (
        count_formal_cell_manifests(study_root),
        int(finalized.is_file() and not finalized.is_symlink()),
    )


def campaign_lock_is_available(path: Path) -> bool:
    if _has_symlink_component(path):
        raise WatchdogError("campaign lock path contains a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    if _has_symlink_component(path):
        raise WatchdogError("campaign lock path contains a symlink")
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o640)
    stream = os.fdopen(descriptor, "a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        return True
    finally:
        stream.close()


def acquire_watchdog_lock(path: Path) -> int:
    """Hold a same-host singleton lock for the watchdog's whole lifetime."""

    path = Path(path)
    if _has_symlink_component(path):
        raise WatchdogError("watchdog lock path contains a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    if _has_symlink_component(path):
        raise WatchdogError("watchdog lock path contains a symlink")
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o640)
    except OSError as error:
        raise WatchdogError("cannot open watchdog singleton lock") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise WatchdogError("watchdog lock is not a regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise WatchdogError("another campaign watchdog is active") from error
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


class Logger(Protocol):
    def emit(self, event: str, **fields: Any) -> None: ...


@dataclass
class TrackedCampaign:
    process: ProcessIdentity
    session: str
    launched_monotonic: float
    progress_at_launch: tuple[int, int]
    kind: str = "campaign"


@dataclass(frozen=True)
class WatchdogDecision:
    action: str
    exit_code: int | None = None


class CampaignWatchdog:
    """One-state-machine watchdog; ``run`` repeatedly calls ``step``."""

    def __init__(
            self, *, repo_root: Path, study_root: Path,
            report_path: Path, stop_sentinel: Path,
            supervisor_pid: int, tmux_session: str,
            inspector: ProcessInspector, tmux: TmuxClient, logger: Logger,
            resource_reader: Callable[[Path], ResourceSnapshot] = read_resources,
            gpu_reader: Callable[[], Mapping[int, int]] = read_gpu_temperatures,
            progress_reader: Callable[[Path], tuple[int, int]] = progress_token,
            closeout_guard_validator: Callable[[], None] = lambda: None,
            pre_reveal_interlock_validator: Callable[[], None] = lambda: None,
            closeout_recovery_command: Sequence[str] | None = None,
            monotonic: Callable[[], float] = time.monotonic,
            sleeper: Callable[[float], None] = time.sleep,
            campaign_poll_seconds: float = 30.0,
            poll_seconds: float = 30.0,
            disk_warn_bytes: int = DEFAULT_DISK_WARN_BYTES,
            disk_stop_bytes: int = DEFAULT_DISK_STOP_BYTES,
            ram_stop_bytes: int = DEFAULT_RAM_STOP_BYTES,
            gpu_warn_celsius: int = 90,
            gpu_stop_celsius: int | None = None,
            max_restarts: int = 3,
            healthy_reset_seconds: float = 300.0,
            orphan_grace_seconds: float = 300.0,
            launch_verify_seconds: float = 30.0,
            term_grace_seconds: float = 10.0,
            kill_grace_seconds: float = 5.0,
            termination_poll_seconds: float = 0.25):
        self.repo_root = Path(os.path.abspath(repo_root))
        self.study_root = Path(os.path.abspath(study_root))
        self.report_path = Path(os.path.abspath(report_path))
        self.stop_sentinel = Path(os.path.abspath(stop_sentinel))
        self.inspector = inspector
        self.tmux = tmux
        self.logger = logger
        self.resource_reader = resource_reader
        self.gpu_reader = gpu_reader
        self.progress_reader = progress_reader
        self.closeout_guard_validator = closeout_guard_validator
        self.pre_reveal_interlock_validator = \
            pre_reveal_interlock_validator
        self.closeout_recovery_command = tuple(
            closeout_recovery_command or closeout_recovery_argv(
                self.repo_root, self.study_root, self.stop_sentinel))
        self.monotonic = monotonic
        self.sleeper = sleeper
        self.campaign_poll_seconds = float(campaign_poll_seconds)
        self.poll_seconds = float(poll_seconds)
        self.disk_warn_bytes = int(disk_warn_bytes)
        self.disk_stop_bytes = int(disk_stop_bytes)
        self.ram_stop_bytes = int(ram_stop_bytes)
        self.gpu_warn_celsius = int(gpu_warn_celsius)
        self.gpu_stop_celsius = (None if gpu_stop_celsius is None
                                 else int(gpu_stop_celsius))
        self.max_restarts = int(max_restarts)
        self.healthy_reset_seconds = float(healthy_reset_seconds)
        self.orphan_grace_seconds = float(orphan_grace_seconds)
        self.launch_verify_seconds = float(launch_verify_seconds)
        self.term_grace_seconds = float(term_grace_seconds)
        self.kill_grace_seconds = float(kill_grace_seconds)
        self.termination_poll_seconds = float(termination_poll_seconds)
        if not (_positive_int(supervisor_pid) and tmux_session
                and self.poll_seconds > 0 and self.campaign_poll_seconds > 0
                and self.disk_warn_bytes > self.disk_stop_bytes > 0
                and self.ram_stop_bytes > 0 and self.max_restarts >= 0
                and self.gpu_warn_celsius > 0
                and (self.gpu_stop_celsius is None
                     or self.gpu_stop_celsius >= self.gpu_warn_celsius)
                and self.healthy_reset_seconds >= 0
                and self.orphan_grace_seconds >= 0
                and self.launch_verify_seconds > 0
                and self.term_grace_seconds >= 0
                and self.kill_grace_seconds >= 0
                and self.termination_poll_seconds > 0
                and self.closeout_recovery_command == closeout_recovery_argv(
                    self.repo_root, self.study_root, self.stop_sentinel)):
            raise WatchdogError("invalid watchdog configuration")
        process = preflight_bootstrap_identity(
            repo_root=self.repo_root, supervisor_pid=supervisor_pid,
            tmux_session=tmux_session,
            campaign_poll_seconds=self.campaign_poll_seconds,
            inspector=inspector, tmux=tmux)
        now = self.monotonic()
        self.tracked: TrackedCampaign | None = TrackedCampaign(
            process=process, session=tmux_session, launched_monotonic=now,
            progress_at_launch=(0, 0))
        self.restart_attempts = 0
        self.restart_candidate: TrackedCampaign | None = None
        self.orphaned_campaign: TrackedCampaign | None = None
        self.orphan_first_seen: float | None = None
        self.orphan_termination_attempted: float | None = None
        self.disk_warned = False
        self.gpu_warned: set[int] = set()
        self.gpu_probe_failure_logged = False
        try:
            self.tracked.progress_at_launch = self.progress_reader(
                self.study_root)
        except Exception as error:
            self._terminate_campaign(self.tracked, emit_event=False)
            raise WatchdogError("cannot establish initial campaign progress") \
                from error
        self.last_progress = self.tracked.progress_at_launch
        self._stop_requested = False
        try:
            self.logger.emit(
                "watchdog-bootstrap", supervisor_pid=process.pid,
                supervisor_start_ticks=process.start_ticks,
                supervisor_uid=process.uid,
                supervisor_session=process.session,
                tmux_session=tmux_session,
                repo_root=str(self.repo_root),
                progress=list(self.tracked.progress_at_launch))
        except Exception as error:
            # A guard that cannot durably record its own activation must not
            # leave the campaign running under the fiction of supervision.
            self._terminate_campaign(self.tracked, emit_event=False)
            raise WatchdogError("cannot initialize durable watchdog log") \
                from error

    def request_stop(self) -> None:
        self._stop_requested = True

    def _stop_is_requested(self) -> bool:
        return self._stop_requested or self.stop_sentinel.exists() \
            or self.stop_sentinel.is_symlink()

    def _tracked_alive(self) -> bool:
        return self.tracked is not None \
            and self.inspector.same_process(self.tracked.process)

    def _session_empty(self, process: ProcessIdentity) -> bool:
        return not self.inspector.session_members(process)

    def _wait_for_session_exit(self, process: ProcessIdentity,
                               seconds: float) -> bool:
        deadline = self.monotonic() + seconds
        while True:
            if self._session_empty(process):
                return True
            if self.monotonic() >= deadline:
                return False
            self.sleeper(min(self.termination_poll_seconds,
                             max(0.0, deadline - self.monotonic())))

    def _terminate_campaign(self, tracked: TrackedCampaign | None, *,
                            reason: str = "unspecified",
                            emit_event: bool = True) -> bool:
        """TERM, verify, then KILL every group in the tracked kernel session."""

        if tracked is None:
            return True
        expected = tracked.process
        pane = self.tmux.pane(tracked.session)
        tmux_kill_requested = False
        # Never act on a dead or PID-mismatched session name: it may have been
        # deleted and recycled since bootstrap.  Kernel-session cleanup below
        # remains identity-bound even when the original leader has exited.
        if pane is not None and not pane.dead \
                and pane.pid == expected.pid \
                and self.inspector.same_process(expected):
            tmux_kill_requested = self.tmux.kill(tracked.session)
        term_requested = self.inspector.signal_session(expected, signal.SIGTERM)
        terminated = self._wait_for_session_exit(
            expected, self.term_grace_seconds)
        kill_requested = False
        if not terminated:
            kill_requested = self.inspector.signal_session(
                expected, signal.SIGKILL)
            terminated = self._wait_for_session_exit(
                expected, self.kill_grace_seconds)
        if emit_event:
            self.logger.emit(
                "tracked-campaign-termination", reason=reason,
                supervisor_pid=expected.pid,
                supervisor_start_ticks=expected.start_ticks,
                supervisor_uid=expected.uid,
                supervisor_session=expected.session,
                tmux_session=tracked.session,
                tmux_kill_requested=tmux_kill_requested,
                term_requested=term_requested,
                kill_requested=kill_requested,
                terminated_and_verified=terminated)
        return terminated

    def _kill_tracked(self, reason: str) -> bool:
        return self._terminate_campaign(
            self.tracked or self.orphaned_campaign or self.restart_candidate,
            reason=reason)

    def _cleanup_created_session(
            self, session: str, candidate: TrackedCampaign | None, *,
            reason: str) -> bool:
        """Terminate every watchdog-created session, even before argv binds."""

        if candidate is not None:
            return self._terminate_campaign(candidate, reason=reason)
        pane = self.tmux.pane(session)
        if pane is None or pane.dead:
            return True
        self.tmux.kill(session)
        deadline = self.monotonic() + self.kill_grace_seconds
        while self.monotonic() <= deadline:
            observed = self.tmux.pane(session)
            if observed is None or observed.dead:
                return True
            self.sleeper(min(
                self.termination_poll_seconds,
                max(0.0, deadline - self.monotonic())))
        return False

    def _pipeline_processes(self) -> tuple[list[ProcessIdentity],
                                           list[ProcessIdentity]]:
        supervisors: list[ProcessIdentity] = []
        orphans: list[ProcessIdentity] = []
        for process in self.inspector.scan():
            if is_exact_supervisor(
                    process, self.repo_root, self.campaign_poll_seconds):
                supervisors.append(process)
            elif is_full_pipeline_process(process, self.repo_root):
                orphans.append(process)
            elif is_exact_closeout_recovery(
                    process, self.repo_root, self.study_root,
                    self.stop_sentinel):
                orphans.append(process)
        return supervisors, orphans

    def _restart(self, now: float) -> bool:
        if self._stop_is_requested():
            return False
        self.restart_attempts += 1
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        session = (
            f"sage-mem-v1-auto-{stamp}-a{self.restart_attempts}")
        log_path = self.study_root / "logs" / (
            f"campaign-supervisor-auto-{stamp}-a{self.restart_attempts}.log")
        command = supervisor_argv(
            self.repo_root, self.campaign_poll_seconds)
        self.logger.emit(
            "restart-launch", attempt=self.restart_attempts,
            tmux_session=session, supervisor_log=str(log_path))
        if not self.tmux.launch(
                session, self.repo_root, command, log_path):
            self.logger.emit(
                "restart-launch-failed", attempt=self.restart_attempts,
                tmux_session=session, reason="tmux-create-or-option-failed")
            return False
        deadline = self.monotonic() + self.launch_verify_seconds
        candidate: TrackedCampaign | None = None
        while self.monotonic() <= deadline:
            pane = self.tmux.pane(session)
            if pane is not None and not pane.dead:
                process = self.inspector.read(pane.pid)
                if process is not None and process.live \
                        and process.uid == os.geteuid() \
                        and process.pid == pane.pid \
                        and process.session == process.pid \
                        and process.pgrp == process.pid:
                    candidate = TrackedCampaign(
                        process=process, session=session,
                        launched_monotonic=now,
                        progress_at_launch=(0, 0))
                    self.restart_candidate = candidate
                    candidate.progress_at_launch = self.progress_reader(
                        self.study_root)
                if candidate is not None and is_exact_supervisor(
                        candidate.process, self.repo_root,
                        self.campaign_poll_seconds):
                    if self._stop_is_requested():
                        terminated = self._terminate_campaign(
                            candidate, reason="stop-during-restart")
                        self.logger.emit(
                            "restart-aborted", attempt=self.restart_attempts,
                            tmux_session=session,
                            reason="stop-during-restart",
                            terminated_and_verified=terminated)
                        self.restart_candidate = None
                        return False
                    self.tracked = candidate
                    self.restart_candidate = None
                    self.last_progress = self.tracked.progress_at_launch
                    self.orphaned_campaign = None
                    self.orphan_first_seen = None
                    self.orphan_termination_attempted = None
                    self.logger.emit(
                        "restart-verified", attempt=self.restart_attempts,
                        supervisor_pid=process.pid,
                        supervisor_start_ticks=process.start_ticks,
                        tmux_session=session,
                        progress=list(self.tracked.progress_at_launch))
                    if self._stop_is_requested():
                        terminated = self._terminate_campaign(
                            self.tracked, reason="stop-during-restart")
                        self.tracked = None
                        self.logger.emit(
                            "restart-aborted", attempt=self.restart_attempts,
                            tmux_session=session,
                            reason="stop-during-restart",
                            terminated_and_verified=terminated)
                        return False
                    return True
            if self._stop_is_requested() and candidate is not None:
                terminated = self._terminate_campaign(
                    candidate, reason="stop-during-restart")
                self.logger.emit(
                    "restart-aborted", attempt=self.restart_attempts,
                    tmux_session=session, reason="stop-during-restart",
                    terminated_and_verified=terminated)
                self.restart_candidate = None
                return False
            self.sleeper(min(1.0, self.poll_seconds))
        terminated = self._cleanup_created_session(
            session, candidate, reason="restart-verification-timeout")
        self.logger.emit(
            "restart-launch-failed", attempt=self.restart_attempts,
            tmux_session=session, reason="exact-process-verification-timeout",
            terminated_and_verified=terminated)
        self.restart_candidate = None
        return False

    def _launch_closeout_recovery(self, now: float) -> bool:
        """Launch and identity-bind the sole registered recovery command."""

        if self._stop_is_requested():
            return False
        self.restart_attempts += 1
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        session = (
            f"sage-mem-v1-closeout-{stamp}-a{self.restart_attempts}")
        log_path = self.study_root / "logs" / (
            f"closeout-recovery-{stamp}-a{self.restart_attempts}.log")
        self.logger.emit(
            "closeout-recovery-launch", attempt=self.restart_attempts,
            tmux_session=session, recovery_log=str(log_path),
            paper_authorization=False)
        if not self.tmux.launch(
                session, self.repo_root, self.closeout_recovery_command,
                log_path):
            self.logger.emit(
                "closeout-recovery-launch-failed",
                attempt=self.restart_attempts, tmux_session=session,
                reason="tmux-create-or-option-failed")
            return False
        deadline = self.monotonic() + self.launch_verify_seconds
        candidate: TrackedCampaign | None = None
        pane: PaneIdentity | None = None
        while self.monotonic() <= deadline:
            pane = self.tmux.pane(session)
            if pane is not None and not pane.dead:
                process = self.inspector.read(pane.pid)
                if process is not None and process.live \
                        and process.uid == os.geteuid() \
                        and process.pid == pane.pid \
                        and process.session == process.pid \
                        and process.pgrp == process.pid:
                    candidate = TrackedCampaign(
                        process=process, session=session,
                        launched_monotonic=now,
                        progress_at_launch=self.progress_reader(
                            self.study_root), kind="recovery")
                    self.restart_candidate = candidate
                if candidate is not None and is_exact_closeout_recovery(
                        candidate.process, self.repo_root, self.study_root,
                        self.stop_sentinel):
                    if self._stop_is_requested():
                        terminated = self._terminate_campaign(
                            candidate, reason="stop-during-closeout-recovery")
                        self.logger.emit(
                            "closeout-recovery-aborted",
                            attempt=self.restart_attempts,
                            tmux_session=session,
                            reason="stop-during-closeout-recovery",
                            terminated_and_verified=terminated)
                        self.restart_candidate = None
                        return False
                    self.tracked = candidate
                    self.restart_candidate = None
                    self.orphaned_campaign = None
                    self.orphan_first_seen = None
                    self.orphan_termination_attempted = None
                    self.logger.emit(
                        "closeout-recovery-verified",
                        attempt=self.restart_attempts,
                        recovery_pid=process.pid,
                        recovery_start_ticks=process.start_ticks,
                        tmux_session=session,
                        paper_authorization=False)
                    return True
            if self._stop_is_requested() and candidate is not None:
                terminated = self._terminate_campaign(
                    candidate, reason="stop-during-closeout-recovery")
                self.logger.emit(
                    "closeout-recovery-aborted",
                    attempt=self.restart_attempts, tmux_session=session,
                    reason="stop-during-closeout-recovery",
                    terminated_and_verified=terminated)
                self.restart_candidate = None
                return False
            self.sleeper(min(1.0, self.poll_seconds))
        terminated = self._cleanup_created_session(
            session, candidate,
            reason="closeout-recovery-verification-timeout")
        self.logger.emit(
            "closeout-recovery-launch-failed", attempt=self.restart_attempts,
            tmux_session=session, reason="exact-process-verification-timeout",
            terminated_and_verified=terminated)
        self.restart_candidate = None
        return False

    def step(self) -> WatchdogDecision:
        now = self.monotonic()
        if self._stop_is_requested():
            terminated_recovery = True
            if self.tracked is not None and self.tracked.kind == "recovery":
                terminated_recovery = self._terminate_campaign(
                    self.tracked, reason="stop-during-closeout-recovery")
            self.logger.emit(
                "watchdog-stop", reason=("signal" if self._stop_requested
                                          else "stop-sentinel"),
                sentinel=str(self.stop_sentinel),
                closeout_recovery_terminated=terminated_recovery)
            return WatchdogDecision("exit", 0)

        try:
            self.closeout_guard_validator()
        except Exception as error:
            terminated = self._kill_tracked("invalid-pre-reveal-custody")
            self.logger.emit(
                "watchdog-fail-closed",
                reason="invalid-pre-reveal-closeout-custody",
                detail=f"{type(error).__name__}: {error}",
                tracked_process_terminated=terminated)
            return WatchdogDecision("exit", 4)
        validate_interlock = (
            self.tracked is not None and self.tracked.kind == "campaign")
        if self.tracked is None:
            validate_interlock = (
                self.progress_reader(self.study_root)[0] < FORMAL_CELL_COUNT)
        if validate_interlock:
            try:
                self.pre_reveal_interlock_validator()
            except Exception as error:
                terminated = self._kill_tracked(
                    "invalid-pre-reveal-interlock")
                self.logger.emit(
                    "watchdog-fail-closed",
                    reason="invalid-pre-reveal-closeout-interlock",
                    detail=f"{type(error).__name__}: {error}",
                    tracked_process_terminated=terminated)
                return WatchdogDecision("exit", 4)

        completion = validate_completion_metadata(
            self.report_path, self.study_root)
        if completion.state == "complete":
            recovery_state = validate_recovery_metadata_state(
                self.study_root, self.report_path)
            if recovery_state.state == "invalid":
                terminated = self._kill_tracked(
                    "invalid-closeout-recovery-state")
                self.logger.emit(
                    "watchdog-fail-closed",
                    reason=recovery_state.reason,
                    tracked_process_terminated=terminated)
                return WatchdogDecision("exit", 4)
            if recovery_state.state == "complete":
                self.logger.emit(
                    "formal-audit-metadata-complete",
                    phase_a_cells_verified=FORMAL_CELL_COUNT,
                    finalized_cells_verified=FORMAL_CELL_COUNT,
                    phase_a_grid_sha256=completion.phase_a_grid_sha256,
                    outcomes_interpreted=False, paper_authorization=False)
                return WatchdogDecision("exit", 0)
            if self.tracked is not None and self.tracked.kind == "recovery" \
                    and self._tracked_alive():
                self.logger.emit(
                    "formal-audit-metadata-published-recovery-pending",
                    recovery_pid=self.tracked.process.pid,
                    paper_authorization=False)
            else:
                terminated = self._kill_tracked(
                    "metadata-report-without-recovery-terminal-state")
                self.logger.emit(
                    "watchdog-fail-closed",
                    reason="metadata-report-without-recovery-terminal-state",
                    tracked_kind=(None if self.tracked is None
                                  else self.tracked.kind),
                    tracked_process_terminated=terminated)
                return WatchdogDecision("exit", 4)
        if completion.state == "invalid":
            terminated = self._kill_tracked("invalid-completion-report")
            self.logger.emit(
                "watchdog-fail-closed", reason=completion.reason,
                tracked_campaign_terminated=terminated)
            return WatchdogDecision("exit", 4)

        resources = self.resource_reader(self.repo_root)
        if resources.disk_available < self.disk_stop_bytes \
                or resources.memory_available < self.ram_stop_bytes:
            terminated = self._kill_tracked("resource-threshold")
            self.logger.emit(
                "watchdog-fail-closed", reason="resource-threshold",
                disk_available=resources.disk_available,
                memory_available=resources.memory_available,
                disk_stop_bytes=self.disk_stop_bytes,
                ram_stop_bytes=self.ram_stop_bytes,
                tracked_campaign_terminated=terminated)
            return WatchdogDecision("exit", 2)
        if resources.disk_available < self.disk_warn_bytes \
                and not self.disk_warned:
            self.disk_warned = True
            self.logger.emit(
                "resource-warning",
                disk_available=resources.disk_available,
                disk_warn_bytes=self.disk_warn_bytes,
                memory_available=resources.memory_available)

        try:
            temperatures = {
                int(index): int(value)
                for index, value in self.gpu_reader().items()}
            self.gpu_probe_failure_logged = False
        except (OSError, subprocess.SubprocessError, WatchdogError,
                TypeError, ValueError) as error:
            temperatures = {}
            if not self.gpu_probe_failure_logged:
                self.gpu_probe_failure_logged = True
                self.logger.emit(
                    "gpu-temperature-unavailable",
                    reason=type(error).__name__)
            if self.gpu_stop_celsius is not None:
                terminated = self._kill_tracked(
                    "gpu-temperature-probe-unavailable")
                self.logger.emit(
                    "watchdog-fail-closed",
                    reason="gpu-temperature-probe-unavailable",
                    detail=type(error).__name__,
                    tracked_campaign_terminated=terminated)
                return WatchdogDecision("exit", 2)
        if self.gpu_stop_celsius is not None and any(
                value >= self.gpu_stop_celsius
                for value in temperatures.values()):
            terminated = self._kill_tracked("gpu-temperature-threshold")
            self.logger.emit(
                "watchdog-fail-closed",
                reason="gpu-temperature-threshold",
                temperatures_celsius=temperatures,
                gpu_stop_celsius=self.gpu_stop_celsius,
                tracked_campaign_terminated=terminated)
            return WatchdogDecision("exit", 2)
        for index, temperature in sorted(temperatures.items()):
            if temperature >= self.gpu_warn_celsius \
                    and index not in self.gpu_warned:
                self.gpu_warned.add(index)
                self.logger.emit(
                    "gpu-temperature-warning", physical_gpu=index,
                    temperature_celsius=temperature,
                    gpu_warn_celsius=self.gpu_warn_celsius,
                    fail_closed_enabled=self.gpu_stop_celsius is not None)

        if self._tracked_alive():
            assert self.tracked is not None
            pane = self.tmux.pane(self.tracked.session)
            if pane is None or pane.dead \
                    or pane.pid != self.tracked.process.pid:
                terminated = self._kill_tracked(
                    "tracked-tmux-session-identity-drift")
                self.logger.emit(
                    "watchdog-fail-closed",
                    reason="tracked-tmux-session-identity-drift",
                    supervisor_pid=self.tracked.process.pid,
                    tmux_session=self.tracked.session,
                    tracked_campaign_terminated=terminated)
                return WatchdogDecision("exit", 4)
            if self.tracked.kind == "recovery" \
                    and not is_exact_closeout_recovery(
                        self.tracked.process, self.repo_root,
                        self.study_root, self.stop_sentinel):
                terminated = self._kill_tracked(
                    "tracked-closeout-recovery-identity-drift")
                self.logger.emit(
                    "watchdog-fail-closed",
                    reason="tracked-closeout-recovery-identity-drift",
                    tracked_process_terminated=terminated)
                return WatchdogDecision("exit", 4)
            current_progress = self.progress_reader(self.study_root)
            if current_progress > self.last_progress:
                self.logger.emit(
                    "campaign-progress", supervisor_pid=self.tracked.process.pid,
                    progress=list(current_progress))
                self.last_progress = current_progress
            if self.tracked.kind == "campaign" \
                    and current_progress > self.tracked.progress_at_launch:
                if self.restart_attempts > 0 and (
                        now - self.tracked.launched_monotonic
                        >= self.healthy_reset_seconds):
                    previous = self.restart_attempts
                    self.restart_attempts = 0
                    self.tracked.progress_at_launch = current_progress
                    self.last_progress = current_progress
                    self.logger.emit(
                        "restart-counter-reset", previous_attempts=previous,
                        reason="healthy-process-with-new-completed-artifact",
                        progress=list(current_progress))
            return WatchdogDecision("wait")

        previous = self.tracked or self.orphaned_campaign
        self.tracked = None
        if previous is not None:
            self.orphaned_campaign = previous
            session_members = self.inspector.session_members(previous.process)
            if session_members:
                if self.orphan_first_seen is None:
                    self.orphan_first_seen = now
                    self.logger.emit(
                        "orphan-session-wait",
                        kernel_session=previous.process.session,
                        pids=[item.pid for item in session_members],
                        grace_seconds=self.orphan_grace_seconds)
                    return WatchdogDecision("wait")
                if now - self.orphan_first_seen < self.orphan_grace_seconds:
                    return WatchdogDecision("wait")
                terminated = self._terminate_campaign(
                    previous, reason="tracked-leader-exited-before-session")
                self.orphan_termination_attempted = now
                if not terminated:
                    self.logger.emit(
                        "watchdog-fail-closed",
                        reason="tracked-kernel-session-resisted-termination",
                        kernel_session=previous.process.session,
                        pids=[item.pid for item in session_members])
                    return WatchdogDecision("exit", 3)
                self.logger.emit(
                    "orphan-session-terminated",
                    reason="tracked-leader-exited-before-session",
                    kernel_session=previous.process.session,
                    pids=[item.pid for item in session_members])
                self.orphaned_campaign = None
                return WatchdogDecision("wait")
            if previous.kind == "recovery":
                pane = self.tmux.pane(previous.session)
                if pane is not None and pane.dead \
                        and pane.dead_status in {0, 2, 75}:
                    reason = (
                        "closeout-recovery-invariant-failure"
                        if pane.dead_status == 2 else
                        "closeout-recovery-stopped-without-watchdog-stop"
                        if pane.dead_status == 75 else
                        "closeout-recovery-exited-without-metadata-report")
                    self.logger.emit(
                        "watchdog-fail-closed", reason=reason,
                        recovery_exit_status=pane.dead_status,
                        tmux_session=previous.session)
                    self.orphaned_campaign = None
                    return WatchdogDecision("exit", 4)
            self.orphaned_campaign = None
        supervisors, orphans = self._pipeline_processes()
        if supervisors:
            self.logger.emit(
                "watchdog-fail-closed",
                reason="untracked-exact-campaign-supervisor",
                supervisor_pids=[item.pid for item in supervisors])
            return WatchdogDecision("exit", 3)
        if orphans:
            # These processes are outside the kernel session we bootstrapped;
            # refusing a restart is safer than signalling an unowned session.
            self.logger.emit(
                "watchdog-fail-closed",
                reason="untracked-pipeline-process-outside-owned-session",
                pids=[item.pid for item in orphans])
            return WatchdogDecision("exit", 3)
        self.orphan_first_seen = None
        self.orphan_termination_attempted = None

        if not campaign_lock_is_available(
                self.study_root / "campaign.lock"):
            self.logger.emit(
                "watchdog-fail-closed",
                reason="campaign-lock-held-without-exact-supervisor")
            return WatchdogDecision("exit", 3)
        if self.restart_attempts >= self.max_restarts:
            self.logger.emit(
                "watchdog-fail-closed", reason="restart-budget-exhausted",
                consecutive_attempts=self.restart_attempts)
            return WatchdogDecision("exit", 3)
        current_progress = self.progress_reader(self.study_root)
        if current_progress[0] == FORMAL_CELL_COUNT:
            self._launch_closeout_recovery(now)
        elif current_progress[0] < FORMAL_CELL_COUNT:
            self._restart(now)
        else:
            self.logger.emit(
                "watchdog-fail-closed",
                reason="formal-cell-count-exceeds-registered-grid",
                progress=list(current_progress))
            return WatchdogDecision("exit", 4)
        if self._stop_is_requested():
            self.logger.emit(
                "watchdog-stop", reason=("signal" if self._stop_requested
                                          else "stop-sentinel"),
                sentinel=str(self.stop_sentinel))
            return WatchdogDecision("exit", 0)
        return WatchdogDecision("wait")

    def run(self) -> int:
        while True:
            try:
                decision = self.step()
            except Exception as error:
                try:
                    terminated = self._kill_tracked(
                        "watchdog-operational-or-invariant-error")
                except Exception:
                    terminated = False
                    candidate = (self.tracked or self.orphaned_campaign
                                 or self.restart_candidate)
                    if candidate is not None:
                        try:
                            self.inspector.signal_session(
                                candidate.process, signal.SIGKILL)
                            terminated = self._wait_for_session_exit(
                                candidate.process, self.kill_grace_seconds)
                        except Exception:
                            terminated = False
                try:
                    self.logger.emit(
                        "watchdog-fail-closed",
                        reason="watchdog-operational-or-invariant-error",
                        detail=f"{type(error).__name__}: {error}",
                        tracked_campaign_terminated=terminated)
                except Exception:
                    # The primary event log itself may be the failed component.
                    # Termination above does not depend on this final receipt.
                    pass
                return 4
            if decision.action == "exit":
                assert decision.exit_code is not None
                return decision.exit_code
            self.sleeper(self.poll_seconds)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supervisor-pid", type=int, required=True)
    parser.add_argument("--tmux-session", required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--study-root", type=Path, default=STUDY_ROOT)
    parser.add_argument(
        "--report", type=Path,
        default=STUDY_ROOT / "formal_audit/report.json")
    parser.add_argument(
        "--event-log", type=Path,
        default=STUDY_ROOT / "logs/campaign-watchdog.jsonl")
    parser.add_argument(
        "--stop-sentinel", type=Path,
        default=STUDY_ROOT / "STOP_CAMPAIGN_WATCHDOG")
    parser.add_argument(
        "--watchdog-lock", type=Path,
        default=STUDY_ROOT / "logs/campaign-watchdog.lock")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--campaign-poll-seconds", type=float, default=30.0)
    parser.add_argument("--disk-warn-gib", type=int, default=250)
    parser.add_argument("--disk-stop-gib", type=int, default=200)
    parser.add_argument("--ram-stop-gib", type=int, default=64)
    parser.add_argument("--gpu-warn-celsius", type=int, default=90)
    parser.add_argument(
        "--gpu-stop-celsius", type=int, default=95,
        help=("fail-closed GPU temperature (production default: 95 C; "
              "warning default: 90 C)"))
    parser.add_argument(
        "--disable-gpu-hard-stop", action="store_true",
        help="explicitly disable the optional GPU temperature hard stop")
    parser.add_argument("--max-restarts", type=int, default=3)
    parser.add_argument("--healthy-reset-seconds", type=float, default=300.0)
    parser.add_argument("--orphan-grace-seconds", type=float, default=300.0)
    parser.add_argument("--launch-verify-seconds", type=float, default=30.0)
    parser.add_argument("--term-grace-seconds", type=float, default=10.0)
    parser.add_argument("--kill-grace-seconds", type=float, default=5.0)
    parser.add_argument("--termination-poll-seconds", type=float, default=0.25)
    return parser.parse_args(argv)


def validate_production_cli_paths(args: argparse.Namespace,
                                  repo_root: Path) -> None:
    """Forbid a decoy study root or alternate watchdog singleton in production."""

    repo = Path(os.path.abspath(repo_root))
    expected = {
        "study_root": STUDY_ROOT,
        "report": STUDY_ROOT / "formal_audit/report.json",
        "event_log": STUDY_ROOT / "logs/campaign-watchdog.jsonl",
        "stop_sentinel": STUDY_ROOT / "STOP_CAMPAIGN_WATCHDOG",
        "watchdog_lock": STUDY_ROOT / "logs/campaign-watchdog.lock",
    }
    for name, wanted in expected.items():
        observed = Path(os.path.abspath(getattr(args, name)))
        if observed != Path(os.path.abspath(wanted)):
            raise WatchdogError(
                f"production watchdog path override is forbidden: {name}")
        try:
            observed.relative_to(repo)
        except ValueError as error:
            raise WatchdogError(
                f"production watchdog path leaves repository: {name}") \
                from error


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    repo = Path(os.path.abspath(args.repo_root))
    if repo != ROOT or not (repo / "scripts/run_sage_mem_v1_campaign.py").is_file():
        raise WatchdogError("repo root differs from watchdog installation")
    validate_production_cli_paths(args, repo)
    lock_descriptor = acquire_watchdog_lock(args.watchdog_lock)
    try:
        inspector = ProcessInspector()
        tmux = TmuxClient()
        preflight_bootstrap_identity(
            repo_root=repo, supervisor_pid=args.supervisor_pid,
            tmux_session=args.tmux_session,
            campaign_poll_seconds=args.campaign_poll_seconds,
            inspector=inspector, tmux=tmux)
        # The unchanged protocol-locked finalizer cannot enforce the new
        # execution-registry custody receipt by itself.  Arm an empty
        # pre-reveal interlock before supervision starts; the exactly tracked
        # recovery process removes it only after re-authentication.
        from scripts.recover_sage_mem_v1_closeout import (
            CloseoutPaths, arm_pre_reveal_interlock,
            seal_pre_reveal_custody, validate_pre_reveal_custody,
            validate_pre_reveal_interlock,
        )

        closeout_paths = CloseoutPaths.production(
            repo, args.study_root, repo / "configs/sage_mem_v1.yaml")
        seal_pre_reveal_custody(closeout_paths)
        arm_pre_reveal_interlock(closeout_paths)

        logger = EventLog(args.event_log)
        watchdog = CampaignWatchdog(
            repo_root=repo, study_root=args.study_root,
            report_path=args.report, stop_sentinel=args.stop_sentinel,
            supervisor_pid=args.supervisor_pid, tmux_session=args.tmux_session,
            inspector=inspector, tmux=tmux, logger=logger,
            closeout_guard_validator=lambda: validate_pre_reveal_custody(
                closeout_paths, verify_raw_content=False),
            pre_reveal_interlock_validator=lambda:
                validate_pre_reveal_interlock(closeout_paths),
            closeout_recovery_command=closeout_recovery_argv(
                repo, args.study_root, args.stop_sentinel),
            campaign_poll_seconds=args.campaign_poll_seconds,
            poll_seconds=args.poll_seconds,
            disk_warn_bytes=args.disk_warn_gib * GIB,
            disk_stop_bytes=args.disk_stop_gib * GIB,
            ram_stop_bytes=args.ram_stop_gib * GIB,
            gpu_warn_celsius=args.gpu_warn_celsius,
            gpu_stop_celsius=(None if args.disable_gpu_hard_stop
                              else args.gpu_stop_celsius),
            max_restarts=args.max_restarts,
            healthy_reset_seconds=args.healthy_reset_seconds,
            orphan_grace_seconds=args.orphan_grace_seconds,
            launch_verify_seconds=args.launch_verify_seconds,
            term_grace_seconds=args.term_grace_seconds,
            kill_grace_seconds=args.kill_grace_seconds,
            termination_poll_seconds=args.termination_poll_seconds)

        def request_stop(_signum: int, _frame: Any) -> None:
            watchdog.request_stop()

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        return watchdog.run()
    finally:
        os.close(lock_descriptor)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except WatchdogError as error:
        print(f"SAGE-Mem watchdog stopped: {error}", file=sys.stderr)
        raise SystemExit(2) from error
