from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import recover_sage_mem_v1_closeout as closeout_module
from scripts.recover_sage_mem_v1_closeout import (
    AUDIT_RELATIVE,
    FINALIZER_RELATIVE,
    FORMAL_AUDIT_RELATIVE,
    RECOVERY_RELATIVE,
    SPEC_RELATIVE,
    WATCHDOG_RELATIVE,
    CloseoutPaths,
    CloseoutRecoveryError,
    CloseoutStopRequested,
    arm_pre_reveal_interlock,
    recover_closeout,
    seal_pre_reveal_custody,
    validate_pre_reveal_custody,
    validate_pre_reveal_interlock,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8")


def _paths(tmp_path: Path) -> CloseoutPaths:
    repo = tmp_path / "repo"
    study = repo / "outputs/sage_mem_v1"
    for relative in (
            FINALIZER_RELATIVE, AUDIT_RELATIVE, FORMAL_AUDIT_RELATIVE,
            SPEC_RELATIVE, RECOVERY_RELATIVE, WATCHDOG_RELATIVE):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"locked:{relative}\n", encoding="utf-8")
    producers = {
        relative: {"sha256": _sha(repo / relative),
                   "size": (repo / relative).stat().st_size}
        for relative in (
            FINALIZER_RELATIVE, AUDIT_RELATIVE, FORMAL_AUDIT_RELATIVE,
            SPEC_RELATIVE)
    }
    _json(study / "protocol_lock.json", {
        "study": "sage-mem-v1", "status": "sealed",
        "protocol_fingerprint": "f" * 64,
        "producer_identities": producers,
    })
    _json(study / "formal_preparation/custody/registry.json", {
        "schema": "synthetic-custody", "status": "sealed"})
    (study / "formal_preparation/custody/vault.bin").write_bytes(
        b"synthetic-label-custody")
    _json(study / "raw_context_phase_a/summary.json", {
        "schema": "synthetic-raw", "status": "complete"})
    (study / "raw_context_phase_a/reference.bin").write_bytes(
        b"synthetic-raw-context")
    execution = study / "formal_preparation/execution_decks"
    _json(execution / "registry.json", {
        "schema": "synthetic-execution", "status": "sealed"})
    (execution / "cohorts/example").mkdir(parents=True)
    (execution / "cohorts/example/cube.bin").write_bytes(b"label-free-cube")
    study.mkdir(parents=True, exist_ok=True)
    return CloseoutPaths.production(repo, study, repo / SPEC_RELATIVE)


@dataclass
class _Grid:
    contract: Any = field(
        default_factory=lambda: SimpleNamespace(total_cells=600))


class FakeFinalizer:
    def __init__(self, paths: CloseoutPaths, *,
                 interrupt: str | None = None,
                 stop_after_finalize: Path | None = None):
        self.paths = paths
        self.interrupt = interrupt
        self.interrupted = False
        self.stop_after_finalize = stop_after_finalize
        self.finalize_calls = 0
        self.label_load_calls = 0
        self.receipt_writer_calls = 0

    def validate_complete_phase_a_grid(self, _root: Path) -> _Grid:
        return _Grid()

    def _validate_raw_context_references(
            self, _root: Path, _grid: _Grid) -> tuple[dict[str, Any], str]:
        return {}, "raw-context-digest"

    def _receipt_value(self) -> dict[str, Any]:
        return {
            "valid": True,
            "execution_registry_sha256": _sha(
                self.paths.execution_registry),
            "label_registry_sha256": _sha(self.paths.label_registry),
            "raw_context_sha256": "raw-context-digest",
            "phase_a_cells": 600,
            "recorded_unix_ns": 999999,
        }

    def _validate_reveal_receipt(
            self, path: Path, _grid: _Grid, registry: Path, *,
            raw_context_sha256: str | None = None,
            execution_deck_registry: Path | None = None) -> dict[str, Any]:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        expected = self._receipt_value()
        observed_time = value.pop("recorded_unix_ns", None)
        expected.pop("recorded_unix_ns")
        if value != expected \
                or not isinstance(observed_time, int) \
                or isinstance(observed_time, bool) \
                or Path(registry).resolve() != self.paths.label_registry.resolve() \
                or raw_context_sha256 != "raw-context-digest" \
                or execution_deck_registry is None \
                or Path(execution_deck_registry).resolve() != \
                self.paths.execution_registry.resolve():
            raise RuntimeError("invalid synthetic reveal receipt")
        return value

    def _record_label_reveal_receipt(
            self, _grid: _Grid, _registry: Path, output: Path, *,
            raw_context_sha256: str | None = None,
            execution_deck_registry: Path | None = None) -> Path:
        del raw_context_sha256, execution_deck_registry
        self.receipt_writer_calls += 1
        receipt = Path(output) / "label_reveal_receipt.json"
        receipt.write_text(
            json.dumps(self._receipt_value(), sort_keys=True,
                       separators=(",", ":")) + "\n",
            encoding="utf-8")
        return receipt

    def finalize_formal_grid(
            self, _phase: Path, _registry: Path, output: Path, *,
            raw_context_root: Path | None = None,
            execution_deck_registry: Path | None = None) -> dict[str, Any]:
        del raw_context_root
        self.finalize_calls += 1
        output = Path(output)
        output.mkdir(parents=True)
        if self.interrupt == "before-receipt" and not self.interrupted:
            self.interrupted = True
            raise RuntimeError("synthetic crash before receipt")
        grid = _Grid()
        self._record_label_reveal_receipt(
            grid, self.paths.label_registry, output,
            raw_context_sha256="raw-context-digest",
            execution_deck_registry=execution_deck_registry)
        if self.interrupt == "after-receipt" and not self.interrupted:
            self.interrupted = True
            raise RuntimeError("synthetic crash after receipt")
        self.label_load_calls += 1
        cells = output / "cells"
        cells.mkdir()
        (cells / "cell-0.bin").write_bytes(b"zero")
        if self.interrupt == "mid-cells" and not self.interrupted:
            self.interrupted = True
            raise RuntimeError("synthetic crash mid cells")
        (cells / "cell-1.bin").write_bytes(b"one")
        summary = {
            "phase_a_cells": 600,
            "phase_a_grid_sha256": "a" * 64,
            "finalized_cells": 600,
            "finalized_cells_sha256": "b" * 64,
        }
        _json(output / "summary.json", summary)
        if self.stop_after_finalize is not None:
            self.stop_after_finalize.touch()
            self.stop_after_finalize = None
        return summary

    def validate_finalized_output(
            self, _phase: Path, _registry: Path, output: Path, *,
            raw_context_root: Path | None = None,
            execution_deck_registry: Path | None = None) -> dict[str, Any]:
        del raw_context_root
        output = Path(output)
        self._validate_reveal_receipt(
            output / "label_reveal_receipt.json", _Grid(),
            self.paths.label_registry,
            raw_context_sha256="raw-context-digest",
            execution_deck_registry=execution_deck_registry)
        if {path.name for path in (output / "cells").iterdir()} != {
                "cell-0.bin", "cell-1.bin"}:
            raise RuntimeError("synthetic finalized cells incomplete")
        return json.loads((output / "summary.json").read_text())


class AuditCounter:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, paths: CloseoutPaths) -> None:
        self.calls += 1
        _json(paths.report_path, {
            "schema": "synthetic-formal-audit", "status": "complete",
            "call": self.calls})


def _ready(tmp_path: Path, *, interrupt: str | None = None,
           stop_after_finalize: Path | None = None
           ) -> tuple[CloseoutPaths, FakeFinalizer, AuditCounter]:
    paths = _paths(tmp_path)
    seal_pre_reveal_custody(paths, now_ns=lambda: 123456)
    arm_pre_reveal_interlock(paths)
    return (paths,
            FakeFinalizer(paths, interrupt=interrupt,
                          stop_after_finalize=stop_after_finalize),
            AuditCounter())


def _recover(paths: CloseoutPaths, finalizer: FakeFinalizer,
             audit: AuditCounter, *, stop: Path | None = None,
             processes: list[int] | None = None) -> dict[str, Any]:
    return recover_closeout(
        paths, stop_sentinel=stop, finalizer=finalizer,
        audit_runner=audit,
        process_scan=lambda *_args, **_kwargs: list(processes or []),
        current_pid=99999)


def test_pre_reveal_receipt_is_mandatory_and_binds_entire_execution_tree(
        tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    receipt = seal_pre_reveal_custody(paths, now_ns=lambda: 123456)
    assert receipt["status"] == "sealed-before-label-reveal"
    assert receipt["paper_authorization"] is False
    assert validate_pre_reveal_custody(paths) == receipt
    paths.execution_registry.write_text("{}", encoding="utf-8")
    with pytest.raises(CloseoutRecoveryError, match="changed"):
        validate_pre_reveal_custody(paths)


def test_pre_reveal_receipt_binds_recovery_and_watchdog_sources(
        tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    seal_pre_reveal_custody(paths, now_ns=lambda: 123456)
    source = paths.repo_root / RECOVERY_RELATIVE
    source.write_text("changed recovery source\n", encoding="utf-8")
    with pytest.raises(CloseoutRecoveryError, match="reliability source"):
        validate_pre_reveal_custody(paths)


def test_duplicate_and_noncanonical_custody_json_are_rejected(
        tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.protocol_lock.write_text(
        '{"study":"sage-mem-v1","study":"sage-mem-v1"}\n',
        encoding="utf-8")
    with pytest.raises(CloseoutRecoveryError, match="duplicate JSON key"):
        seal_pre_reveal_custody(paths, now_ns=lambda: 123456)

    paths = _paths(tmp_path / "second")
    receipt = seal_pre_reveal_custody(paths, now_ns=lambda: 123456)
    paths.pre_reveal_receipt.chmod(0o640)
    paths.pre_reveal_receipt.write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n",
        encoding="utf-8")
    with pytest.raises(CloseoutRecoveryError, match="not canonical JSON"):
        validate_pre_reveal_custody(paths)


@pytest.mark.parametrize("mutation", ["missing", "changed"])
def test_missing_or_changed_execution_registry_fails_before_label_reveal(
        tmp_path: Path, mutation: str) -> None:
    paths, finalizer, audit = _ready(tmp_path)
    if mutation == "missing":
        paths.execution_registry.unlink()
    else:
        paths.execution_registry.chmod(0o640)
        paths.execution_registry.write_text("{}", encoding="utf-8")
    with pytest.raises(CloseoutRecoveryError):
        _recover(paths, finalizer, audit)
    assert finalizer.receipt_writer_calls == 0
    assert finalizer.label_load_calls == 0
    assert audit.calls == 0


def test_changed_execution_artifact_fails_before_label_reveal(
        tmp_path: Path) -> None:
    paths, finalizer, audit = _ready(tmp_path)
    cube = paths.execution_root / "cohorts/example/cube.bin"
    cube.write_bytes(b"changed-label-free-cube")
    with pytest.raises(CloseoutRecoveryError, match="execution-deck tree"):
        _recover(paths, finalizer, audit)
    assert finalizer.receipt_writer_calls == 0
    assert finalizer.label_load_calls == 0


@pytest.mark.parametrize("tree", ["custody", "raw"])
def test_changed_custody_or_raw_artifact_fails_before_label_reveal(
        tmp_path: Path, tree: str) -> None:
    paths, finalizer, audit = _ready(tmp_path)
    artifact = (paths.label_registry.parent / "vault.bin"
                if tree == "custody"
                else paths.raw_context_root / "reference.bin")
    info = artifact.stat()
    original = artifact.read_bytes()
    artifact.write_bytes(bytes((byte ^ 1) for byte in original))
    os.utime(artifact, ns=(info.st_atime_ns, info.st_mtime_ns))
    with pytest.raises(CloseoutRecoveryError, match="tree changed"):
        _recover(paths, finalizer, audit)
    assert finalizer.receipt_writer_calls == 0
    assert finalizer.label_load_calls == 0


def test_interruption_before_receipt_archives_only_empty_pre_reveal_staging(
        tmp_path: Path) -> None:
    paths, finalizer, audit = _ready(tmp_path, interrupt="before-receipt")
    with pytest.raises(RuntimeError, match="before receipt"):
        _recover(paths, finalizer, audit)
    assert not paths.reveal_anchor.exists()
    staging = list(paths.study_root.glob(
        ".formal_finalized.recovery-staging-*"))
    assert len(staging) == 1 and not any(staging[0].iterdir())
    result = _recover(paths, finalizer, audit)
    assert result["mode"] == "pre-reveal-fresh-restart"
    assert paths.final_root.is_dir()
    assert paths.reveal_anchor.read_bytes() == (
        paths.final_root / "label_reveal_receipt.json").read_bytes()
    assert audit.calls == 1


def test_nonempty_staging_without_receipt_is_not_archived_or_revealed(
        tmp_path: Path) -> None:
    paths, finalizer, audit = _ready(tmp_path)
    stale = paths.study_root / ".formal_finalized.recovery-staging-forged"
    stale.mkdir()
    (stale / "unexpected.bin").write_bytes(b"unsafe")
    with pytest.raises(CloseoutRecoveryError, match="no valid reveal receipt"):
        _recover(paths, finalizer, audit)
    assert stale.is_dir()
    assert finalizer.finalize_calls == 0
    assert finalizer.label_load_calls == 0


@pytest.mark.parametrize("temporary_name", [
    ".label_reveal_receipt.json.partial-123-deadbeef",
    ".label_reveal_receipt.json.random.tmp",
])
def test_receipt_writer_temporary_before_reveal_is_safely_archived(
        tmp_path: Path, temporary_name: str) -> None:
    paths, finalizer, audit = _ready(tmp_path)
    stale = paths.study_root / ".formal_finalized.recovery-staging-crashed"
    stale.mkdir()
    (stale / temporary_name).write_bytes(b"partial receipt bytes")
    result = _recover(paths, finalizer, audit)
    assert result["mode"] == "pre-reveal-fresh-restart"
    assert finalizer.label_load_calls == 1
    assert any(paths.archive_root.glob("interrupted-staging-*"))


def test_interruption_after_receipt_preserves_receipt_bytes_exactly(
        tmp_path: Path) -> None:
    paths, finalizer, audit = _ready(tmp_path, interrupt="after-receipt")
    with pytest.raises(RuntimeError, match="after receipt"):
        _recover(paths, finalizer, audit)
    staging_receipt = next(paths.study_root.glob(
        ".formal_finalized.recovery-staging-*/label_reveal_receipt.json"))
    original = staging_receipt.read_bytes()
    result = _recover(paths, finalizer, audit)
    assert result["mode"] == "post-reveal-reconstruction"
    assert paths.reveal_anchor.read_bytes() == original
    assert (paths.final_root / "label_reveal_receipt.json").read_bytes() == original
    assert audit.calls == 1


def test_interruption_mid_cells_reconstructs_fresh_validated_staging(
        tmp_path: Path) -> None:
    paths, finalizer, audit = _ready(tmp_path, interrupt="mid-cells")
    with pytest.raises(RuntimeError, match="mid cells"):
        _recover(paths, finalizer, audit)
    assert next(paths.study_root.glob(
        ".formal_finalized.recovery-staging-*/cells/cell-0.bin")).is_file()
    result = _recover(paths, finalizer, audit)
    assert result["mode"] == "post-reveal-reconstruction"
    assert {path.name for path in (paths.final_root / "cells").iterdir()} == {
        "cell-0.bin", "cell-1.bin"}
    assert any(paths.archive_root.glob("interrupted-staging-*"))


def test_invalid_reveal_receipt_stops_without_archiving_partial_root(
        tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    seal_pre_reveal_custody(paths, now_ns=lambda: 123456)
    paths.final_root.mkdir()
    bad = paths.final_root / "label_reveal_receipt.json"
    bad.write_text('{"valid":false}\n', encoding="utf-8")
    finalizer = FakeFinalizer(paths)
    audit = AuditCounter()
    with pytest.raises(CloseoutRecoveryError, match="invalid"):
        _recover(paths, finalizer, audit)
    assert bad.is_file()
    assert not paths.archive_root.exists()
    assert finalizer.finalize_calls == 0


def test_reveal_receipt_must_strictly_postdate_custody_seal(
        tmp_path: Path) -> None:
    paths, finalizer, audit = _ready(tmp_path)
    paths.final_root.chmod(0o700)
    value = finalizer._receipt_value()
    value["recorded_unix_ns"] = 123456
    _json(paths.final_root / "label_reveal_receipt.json", value)
    with pytest.raises(CloseoutRecoveryError, match="postdate custody"):
        _recover(paths, finalizer, audit)
    assert not paths.archive_root.exists()
    assert finalizer.finalize_calls == 0


def test_conflicting_reveal_receipts_stop_without_archiving(
        tmp_path: Path) -> None:
    paths, finalizer, audit = _ready(tmp_path)
    paths.final_root.chmod(0o700)
    good = paths.final_root / "label_reveal_receipt.json"
    good.write_text(
        json.dumps(finalizer._receipt_value(), sort_keys=True,
                   separators=(",", ":")) + "\n",
        encoding="utf-8")
    paths.reveal_anchor.parent.mkdir(parents=True, exist_ok=True)
    paths.reveal_anchor.write_text('{"valid":false}\n', encoding="utf-8")
    with pytest.raises(CloseoutRecoveryError, match="conflicting"):
        _recover(paths, finalizer, audit)
    assert good.is_file()
    assert not paths.archive_root.exists()


def test_duplicate_key_reveal_receipt_is_invalid_and_not_archived(
        tmp_path: Path) -> None:
    paths, finalizer, audit = _ready(tmp_path)
    paths.final_root.chmod(0o700)
    receipt = paths.final_root / "label_reveal_receipt.json"
    receipt.write_text('{"valid":true,"valid":false}\n', encoding="utf-8")
    with pytest.raises(CloseoutRecoveryError, match="duplicate JSON key"):
        _recover(paths, finalizer, audit)
    assert receipt.is_file()
    assert not paths.archive_root.exists()


def test_symlinked_registry_tree_and_output_root_are_rejected(
        tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    real_registry = paths.execution_registry.with_name("registry.real.json")
    paths.execution_registry.rename(real_registry)
    paths.execution_registry.symlink_to(real_registry.name)
    with pytest.raises(CloseoutRecoveryError, match="symlink"):
        seal_pre_reveal_custody(paths, now_ns=lambda: 123456)

    paths.execution_registry.unlink()
    real_registry.rename(paths.execution_registry)
    seal_pre_reveal_custody(paths, now_ns=lambda: 123456)
    target = paths.study_root / "outside-final-root"
    target.mkdir()
    paths.final_root.symlink_to(target, target_is_directory=True)
    with pytest.raises(CloseoutRecoveryError, match="symlink"):
        arm_pre_reveal_interlock(paths)


def test_symlinked_receipt_or_archive_root_and_outside_layout_are_rejected(
        tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    outside_receipts = tmp_path / "outside-receipts"
    outside_receipts.mkdir()
    paths.receipt_root.parent.mkdir(parents=True, exist_ok=True)
    paths.receipt_root.symlink_to(outside_receipts, target_is_directory=True)
    with pytest.raises(CloseoutRecoveryError, match="symlink"):
        seal_pre_reveal_custody(paths, now_ns=lambda: 123456)

    paths = _paths(tmp_path / "archive")
    seal_pre_reveal_custody(paths, now_ns=lambda: 123456)
    arm_pre_reveal_interlock(paths)
    outside_archive = tmp_path / "outside-archive"
    outside_archive.mkdir()
    paths.archive_root.symlink_to(outside_archive, target_is_directory=True)
    with pytest.raises(CloseoutRecoveryError, match="symlink"):
        _recover(paths, FakeFinalizer(paths), AuditCounter())

    forged = replace(
        paths, archive_root=tmp_path / "forged-outside-archive")
    with pytest.raises(CloseoutRecoveryError, match="registered layout"):
        validate_pre_reveal_custody(forged)


@pytest.mark.parametrize("root_state", ["absent", "empty"])
def test_anchor_only_restart_still_arms_closeout_interlock(
        tmp_path: Path, root_state: str) -> None:
    paths, finalizer, _ = _ready(tmp_path)
    paths.reveal_anchor.parent.mkdir(parents=True, exist_ok=True)
    _json(paths.reveal_anchor, finalizer._receipt_value())
    paths.final_root.chmod(0o700)
    paths.final_root.rmdir()
    if root_state == "empty":
        paths.final_root.mkdir(mode=0o700)
    arm_pre_reveal_interlock(paths)
    assert paths.final_root.is_dir()
    assert (paths.final_root.stat().st_mode & 0o777) == 0o500


@pytest.mark.parametrize("mutation", ["mode", "content"])
def test_pre_reveal_interlock_mutation_is_detected(
        tmp_path: Path, mutation: str) -> None:
    paths, _, _ = _ready(tmp_path)
    if mutation == "mode":
        paths.final_root.chmod(0o700)
    else:
        paths.final_root.chmod(0o700)
        (paths.final_root / "unexpected").write_bytes(b"bypass")
        paths.final_root.chmod(0o500)
    with pytest.raises(CloseoutRecoveryError, match="interlock"):
        validate_pre_reveal_interlock(paths)


def test_recovery_is_idempotent_and_does_not_refinalize_complete_root(
        tmp_path: Path) -> None:
    paths, finalizer, audit = _ready(tmp_path)
    first = _recover(paths, finalizer, audit)
    original_receipt = paths.recovery_receipt.read_bytes()
    second = _recover(paths, finalizer, audit)
    assert first["mode"] == "pre-reveal-fresh-restart"
    assert second["mode"] == "already-complete"
    assert finalizer.finalize_calls == 1
    assert audit.calls == 2
    assert paths.recovery_receipt.read_bytes() == original_receipt


def test_stop_after_reconstruction_leaves_recoverable_staging(
        tmp_path: Path) -> None:
    paths, _, _ = _ready(tmp_path)
    stop = paths.study_root / "STOP"
    finalizer = FakeFinalizer(paths, stop_after_finalize=stop)
    audit = AuditCounter()
    with pytest.raises(CloseoutStopRequested):
        _recover(paths, finalizer, audit, stop=stop)
    assert not paths.final_root.exists()
    assert list(paths.study_root.glob(
        ".formal_finalized.recovery-staging-*"))
    stop.unlink()
    result = _recover(paths, finalizer, audit, stop=stop)
    assert result["mode"] == "post-reveal-reconstruction"
    assert audit.calls == 1


def test_preexisting_stop_sentinel_prevents_any_recovery_mutation(
        tmp_path: Path) -> None:
    paths, finalizer, audit = _ready(tmp_path)
    stop = paths.study_root / "STOP"
    stop.touch()
    before_mode = paths.final_root.stat().st_mode
    with pytest.raises(CloseoutStopRequested):
        _recover(paths, finalizer, audit, stop=stop)
    assert paths.final_root.is_dir() and not any(paths.final_root.iterdir())
    assert paths.final_root.stat().st_mode == before_mode
    assert finalizer.finalize_calls == 0


def test_recovery_requires_no_pipeline_process_before_lock(
        tmp_path: Path) -> None:
    paths, finalizer, audit = _ready(tmp_path)
    with pytest.raises(CloseoutRecoveryError, match="pipeline process"):
        _recover(paths, finalizer, audit, processes=[1234])
    assert finalizer.finalize_calls == 0
    assert paths.final_root.is_dir()


def test_production_shaped_paths_forbid_all_injected_hooks(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(closeout_module, "ROOT", paths.repo_root)
    monkeypatch.setattr(closeout_module, "STUDY_ROOT", paths.study_root)
    monkeypatch.setattr(closeout_module, "SPEC_PATH", paths.spec_path)
    with pytest.raises(CloseoutRecoveryError, match="forbids injected"):
        recover_closeout(
            paths, finalizer=FakeFinalizer(paths),
            audit_runner=AuditCounter(),
            process_scan=lambda *_args, **_kwargs: [], current_pid=123)
    assert not paths.recovery_receipt.exists()


def test_recovery_does_not_modify_any_protocol_locked_source(
        tmp_path: Path) -> None:
    paths, finalizer, audit = _ready(tmp_path)
    before = {
        relative: _sha(paths.repo_root / relative)
        for relative in (
            FINALIZER_RELATIVE, AUDIT_RELATIVE, FORMAL_AUDIT_RELATIVE,
            SPEC_RELATIVE)
    }
    _recover(paths, finalizer, audit)
    after = {relative: _sha(paths.repo_root / relative)
             for relative in before}
    assert after == before
