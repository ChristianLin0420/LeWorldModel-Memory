from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import signal

import pytest

from scripts.watch_sage_mem_v1_campaign import (
    ARMS,
    COHORTS,
    FORMAL_CONFIRMATION,
    GIB,
    FINALIZER_SCHEMA,
    REPORT_SCHEMA,
    REPORT_STAGE,
    REPORT_STATUS,
    REPORT_STUDY,
    CampaignWatchdog,
    CompletionCheck,
    EventLog,
    PaneIdentity,
    ProcessIdentity,
    ResourceSnapshot,
    WatchdogError,
    WatchdogDecision,
    acquire_watchdog_lock,
    is_exact_full_launcher,
    is_exact_full_worker,
    is_exact_supervisor,
    supervisor_argv,
    validate_completion_report,
)


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


class FakeLogger:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def emit(self, event: str, **fields: object) -> None:
        self.events.append({"event": event, **fields})


class FakeInspector:
    def __init__(self, processes: list[ProcessIdentity], *,
                 resist_term: bool = False) -> None:
        self.processes = {process.pid: process for process in processes}
        self.terminated: list[int] = []
        self.signals: list[int] = []
        self.resist_term = resist_term

    def read(self, pid: int) -> ProcessIdentity | None:
        return self.processes.get(pid)

    def scan(self) -> list[ProcessIdentity]:
        return [process for process in self.processes.values()
                if process.live]

    def same_process(self, expected: ProcessIdentity) -> bool:
        return self.processes.get(expected.pid) == expected and expected.live

    def session_members(self, expected: ProcessIdentity) \
            -> list[ProcessIdentity]:
        return [process for process in self.processes.values()
                if process.live and process.uid == expected.uid
                and process.session == expected.session]

    def signal_session(self, expected: ProcessIdentity, signum: int) -> bool:
        self.signals.append(signum)
        if signum == signal.SIGTERM and self.resist_term:
            return True
        members = self.session_members(expected)
        self.terminated.extend(process.pid for process in members)
        for process in members:
            self.processes.pop(process.pid, None)
        return True


class FakeTmux:
    def __init__(self, inspector: FakeInspector, initial_session: str,
                 initial_pid: int, *, launch_identity_valid: bool = True) -> None:
        self.inspector = inspector
        self.panes = {
            initial_session: PaneIdentity(initial_pid, False, None),
        }
        self.launch_identity_valid = launch_identity_valid
        self.launched: list[str] = []
        self.killed: list[str] = []
        self.next_pid = 9000

    def pane(self, session: str) -> PaneIdentity | None:
        return self.panes.get(session)

    def kill(self, session: str) -> bool:
        pane = self.panes.pop(session, None)
        self.killed.append(session)
        return pane is not None

    def launch(self, session: str, repo_root: Path, command: tuple[str, ...],
               _log_path: Path) -> bool:
        self.launched.append(session)
        pid = self.next_pid
        self.next_pid += 1
        argv = command if self.launch_identity_valid else (*command, "--bad")
        process = ProcessIdentity(
            pid=pid, uid=os.geteuid(), ppid=1, pgrp=pid, session=pid,
            start_ticks=pid * 10, state="S", cwd=repo_root,
            argv=tuple(argv))
        self.inspector.processes[pid] = process
        self.panes[session] = PaneIdentity(pid, False, None)
        return True


def _supervisor(repo: Path, pid: int = 100, start: int = 1234) \
        -> ProcessIdentity:
    return ProcessIdentity(
        pid=pid, uid=os.geteuid(), ppid=1, pgrp=pid, session=pid,
        start_ticks=start, state="S", cwd=repo,
        argv=(".venv/bin/python", *supervisor_argv(repo)[1:]))


def _full_launcher(repo: Path, pid: int = 200) -> ProcessIdentity:
    return ProcessIdentity(
        pid=pid, uid=os.geteuid(), ppid=100, pgrp=100, session=100,
        start_ticks=2222, state="S", cwd=repo,
        argv=(
            str(repo / ".venv/bin/python"),
            "scripts/launch_sage_mem_v1.py",
            "--spec", str(repo / "configs/sage_mem_v1.yaml"),
            "--execute", "--stage", "full", "--resume",
            "--formal-confirmation", FORMAL_CONFIRMATION,
        ))


def _full_worker(repo: Path, pid: int = 300) -> ProcessIdentity:
    return ProcessIdentity(
        pid=pid, uid=os.geteuid(), ppid=200, pgrp=100, session=100,
        start_ticks=3333, state="R", cwd=repo,
        argv=(
            str(repo / ".venv/bin/python"),
            "scripts/run_sage_mem_v1.py", "--stage", "full",
            "--spec", str(repo / "configs/sage_mem_v1.yaml"),
            "--execute", "--cohort", COHORTS[0], "--arm", ARMS[1],
            "--seed", "3", "--formal-confirmation", FORMAL_CONFIRMATION,
            "--resume",
        ))


def _valid_report() -> dict[str, object]:
    return {
        "schema": REPORT_SCHEMA,
        "study": REPORT_STUDY,
        "stage": REPORT_STAGE,
        "status": REPORT_STATUS,
        "phase_a_cells_verified": 600,
        "finalized_cells_verified": 600,
        "phase_a_grid_sha256": "a" * 64,
        "identity_ledger_sha256": "b" * 64,
        "comparators_verified": 5,
        "resources_verified": 600,
        "raw_context_references_verified": 50,
        "bootstrap_draws_per_contrast": 20000,
        # The monitor validates presence, but deliberately ignores all
        # outcome-bearing values.
        "cohorts": {"must_not_be_interpreted": {"score": 999}},
        "execution_program": {},
        "prior_can_substitute_for_host_output": False,
        "per_age_claims_only": True,
        "pooled_cross_host_score_computed": False,
        "universal_success_claim_permitted": False,
    }


def _write_completion_bundle(study: Path) -> Path:
    for cohort in COHORTS:
        for arm in ARMS:
            for seed in range(10):
                phase_path = (study / "cells" / cohort / arm
                              / f"seed-{seed}" / "manifest.json")
                phase_path.parent.mkdir(parents=True, exist_ok=True)
                phase_path.write_text(json.dumps({
                    "schema": "sage_mem_v1_phase_a_cell_v1",
                    "study": REPORT_STUDY,
                    "stage": "formal-phase-a",
                    "status": "complete-label-free",
                    "cohort": cohort,
                    "arm": arm,
                    "seed": seed,
                }), encoding="utf-8")
                final_path = (study / "formal_finalized" / "cells" / cohort
                              / arm / f"seed-{seed}" / "manifest.json")
                final_path.parent.mkdir(parents=True, exist_ok=True)
                phase_hash = hashlib.sha256(phase_path.read_bytes()).hexdigest()
                final_path.write_text(json.dumps({
                    "schema": FINALIZER_SCHEMA,
                    "study": REPORT_STUDY,
                    "stage": "formal-finalized",
                    "status": REPORT_STATUS,
                    "cohort": cohort,
                    "arm": arm,
                    "seed": seed,
                    "phase_a_grid_sha256": "a" * 64,
                    "phase_a_manifest_sha256": phase_hash,
                }), encoding="utf-8")
    summary = {
        "schema": FINALIZER_SCHEMA,
        "study": REPORT_STUDY,
        "stage": "formal-finalizer",
        "status": REPORT_STATUS,
        "phase_a_cells": 600,
        "phase_a_grid_sha256": "a" * 64,
        "label_reveal_receipt_sha256": "c" * 64,
        "label_registry_sha256": "d" * 64,
        "development_outcomes_read": False,
        "per_age_results_preserved": True,
        "pointmaze_x4_native_clustering_preserved": True,
        "raw_context_reference": {},
        "execution_decks": {},
        "finalized_cells_sha256": "e" * 64,
        "finalized_cells": 600,
    }
    summary_path = study / "formal_finalized" / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    report = study / "formal_audit" / "report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(_valid_report()), encoding="utf-8")
    return report


@pytest.mark.parametrize(("field", "replacement"), [
    ("schema", "old-schema"),
    ("study", "another-study"),
    ("stage", "audit"),
    ("status", "partial"),
    ("phase_a_cells_verified", 599),
    ("phase_a_cells_verified", True),
    ("finalized_cells_verified", 599),
    ("phase_a_grid_sha256", "short"),
    ("identity_ledger_sha256", None),
])
def test_completion_report_requires_exact_identity_and_counts(
        tmp_path: Path, field: str, replacement: object) -> None:
    study = tmp_path / "study"
    report = _write_completion_bundle(study)
    assert validate_completion_report(report, study).state == "complete"

    value = _valid_report()
    value[field] = replacement
    report.write_text(json.dumps(value), encoding="utf-8")
    result = validate_completion_report(report, study)
    assert result.state == "invalid"


def test_completion_report_rejects_symlink_and_accepts_absence(
        tmp_path: Path) -> None:
    absent = tmp_path / "absent.json"
    assert validate_completion_report(absent, tmp_path) == CompletionCheck("absent")
    source = tmp_path / "source.json"
    source.write_text(json.dumps(_valid_report()), encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(source)
    assert validate_completion_report(link, tmp_path).state == "invalid"


def test_process_predicates_require_exact_repo_and_argv(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    supervisor = _supervisor(repo)
    launcher = _full_launcher(repo)
    worker = _full_worker(repo)
    assert is_exact_supervisor(supervisor, repo)
    assert is_exact_full_launcher(launcher, repo)
    assert is_exact_full_worker(worker, repo)
    assert not is_exact_supervisor(replace(supervisor, cwd=tmp_path), repo)
    assert not is_exact_full_launcher(
        replace(launcher, argv=(*launcher.argv, "--extra")), repo)
    assert not is_exact_full_worker(
        replace(worker, argv=(*worker.argv[:-1], "--not-resume")), repo)
    assert not is_exact_supervisor(
        replace(supervisor, uid=os.geteuid() + 1), repo)


def test_completion_report_cross_binds_summary_and_exact_inventories(
        tmp_path: Path) -> None:
    study = tmp_path / "study"
    report = _write_completion_bundle(study)
    summary_path = study / "formal_finalized" / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["phase_a_grid_sha256"] = "9" * 64
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    result = validate_completion_report(report, study)
    assert result == CompletionCheck(
        "invalid", "report-finalizer-grid-hash-mismatch")

    summary["phase_a_grid_sha256"] = "a" * 64
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    extra = study / "cells" / "unregistered" / "manifest.json"
    extra.parent.mkdir(parents=True)
    extra.write_text("{}", encoding="utf-8")
    assert validate_completion_report(report, study) == CompletionCheck(
        "invalid", "phase-a-manifest-inventory-mismatch")


def test_completion_report_rejects_extra_report_metadata(tmp_path: Path) -> None:
    study = tmp_path / "study"
    report = _write_completion_bundle(study)
    value = json.loads(report.read_text())
    value["unexpected"] = "stale-or-unregistered"
    report.write_text(json.dumps(value), encoding="utf-8")
    assert validate_completion_report(report, study) == CompletionCheck(
        "invalid", "report-top-level-keys-mismatch")


def test_completion_report_rejects_cell_changed_after_finalization(
        tmp_path: Path) -> None:
    study = tmp_path / "study"
    report = _write_completion_bundle(study)
    phase = (study / "cells" / COHORTS[0] / ARMS[0] / "seed-0"
             / "manifest.json")
    value = json.loads(phase.read_text())
    value["post_finalization_tamper"] = True
    phase.write_text(json.dumps(value), encoding="utf-8")
    assert validate_completion_report(report, study) == CompletionCheck(
        "invalid", "finalized-manifest-cross-binding-mismatch")


def _watchdog(
        tmp_path: Path, *, resources: list[ResourceSnapshot] | None = None,
        temperatures: dict[int, int] | None = None,
        launch_identity_valid: bool = True,
        healthy_reset_seconds: float = 10.0,
        orphan_grace_seconds: float = 10.0,
        gpu_stop_celsius: int | None = None,
) -> tuple[CampaignWatchdog, FakeInspector, FakeTmux, FakeLogger,
           FakeClock, list[tuple[int, int]]]:
    repo = tmp_path / "repo"
    study = repo / "outputs/sage_mem_v1"
    study.mkdir(parents=True)
    process = _supervisor(repo)
    inspector = FakeInspector([process])
    tmux = FakeTmux(
        inspector, "initial", process.pid,
        launch_identity_valid=launch_identity_valid)
    logger = FakeLogger()
    clock = FakeClock()
    resource_values = resources or [ResourceSnapshot(400 * GIB, 400 * GIB)]
    progress = [(10, 0)]

    def resource_reader(_path: Path) -> ResourceSnapshot:
        return resource_values[-1]

    watchdog = CampaignWatchdog(
        repo_root=repo, study_root=study,
        report_path=study / "formal_audit/report.json",
        stop_sentinel=study / "STOP", supervisor_pid=process.pid,
        tmux_session="initial", inspector=inspector, tmux=tmux,
        logger=logger, resource_reader=resource_reader,
        gpu_reader=lambda: temperatures or {0: 70, 1: 71, 2: 72},
        progress_reader=lambda _path: progress[-1], monotonic=clock,
        sleeper=clock.sleep, poll_seconds=1,
        disk_warn_bytes=250 * GIB, disk_stop_bytes=200 * GIB,
        ram_stop_bytes=64 * GIB, gpu_warn_celsius=90,
        gpu_stop_celsius=gpu_stop_celsius,
        healthy_reset_seconds=healthy_reset_seconds,
        orphan_grace_seconds=orphan_grace_seconds,
        launch_verify_seconds=3)
    return watchdog, inspector, tmux, logger, clock, progress


def _make_tracked_process_disappear(
        watchdog: CampaignWatchdog, inspector: FakeInspector,
        tmux: FakeTmux) -> None:
    assert watchdog.tracked is not None
    pid = watchdog.tracked.process.pid
    session = watchdog.tracked.session
    inspector.processes.pop(pid)
    tmux.panes[session] = PaneIdentity(pid, True, 2)


def test_stop_sentinel_exits_without_killing_or_relaunching(
        tmp_path: Path) -> None:
    watchdog, _, tmux, _, _, _ = _watchdog(tmp_path)
    watchdog.stop_sentinel.touch()
    assert watchdog.step() == WatchdogDecision("exit", 0)
    assert tmux.killed == []
    assert tmux.launched == []


def test_resource_guard_kills_session_created_by_restart(tmp_path: Path) -> None:
    resources = [ResourceSnapshot(400 * GIB, 400 * GIB)]
    watchdog, inspector, tmux, _, _, _ = _watchdog(
        tmp_path, resources=resources)
    _make_tracked_process_disappear(watchdog, inspector, tmux)
    assert watchdog.step().action == "wait"
    assert watchdog.tracked is not None
    restarted_session = watchdog.tracked.session
    assert restarted_session != "initial"

    resources.append(ResourceSnapshot(199 * GIB, 400 * GIB))
    assert watchdog.step() == WatchdogDecision("exit", 2)
    assert restarted_session in tmux.killed


def test_resource_guard_terminates_and_verifies_all_session_groups(
        tmp_path: Path) -> None:
    resources = [ResourceSnapshot(199 * GIB, 400 * GIB)]
    watchdog, inspector, tmux, _, _, _ = _watchdog(
        tmp_path, resources=resources)
    descendant = ProcessIdentity(
        pid=444, uid=os.geteuid(), ppid=100, pgrp=444, session=100,
        start_ticks=4444, state="S", cwd=watchdog.repo_root,
        argv=("unregistered-finalizer-helper",))
    inspector.processes[descendant.pid] = descendant
    assert watchdog.step() == WatchdogDecision("exit", 2)
    assert set(inspector.terminated) == {100, 444}
    assert inspector.session_members(watchdog.tracked.process) == []
    assert inspector.signals == [signal.SIGTERM]
    assert tmux.killed == ["initial"]


def test_resource_guard_escalates_term_resistant_session_to_kill(
        tmp_path: Path) -> None:
    watchdog, inspector, _, _, _, _ = _watchdog(
        tmp_path, resources=[ResourceSnapshot(199 * GIB, 400 * GIB)])
    inspector.resist_term = True
    assert watchdog.step() == WatchdogDecision("exit", 2)
    assert inspector.signals == [signal.SIGTERM, signal.SIGKILL]
    assert inspector.processes == {}


def test_termination_never_kills_dead_pid_mismatched_tmux_session(
        tmp_path: Path) -> None:
    watchdog, inspector, tmux, _, _, _ = _watchdog(
        tmp_path, resources=[ResourceSnapshot(199 * GIB, 400 * GIB)])
    tmux.panes["initial"] = PaneIdentity(999, True, 2)
    assert watchdog.step() == WatchdogDecision("exit", 2)
    assert tmux.killed == []
    assert inspector.processes == {}


def test_verified_orphan_is_terminated_before_safe_relaunch(
        tmp_path: Path) -> None:
    watchdog, inspector, tmux, logger, clock, _ = _watchdog(
        tmp_path, orphan_grace_seconds=5)
    _make_tracked_process_disappear(watchdog, inspector, tmux)
    orphan = _full_worker(watchdog.repo_root)
    inspector.processes[orphan.pid] = orphan
    assert watchdog.step() == WatchdogDecision("wait")
    assert tmux.launched == []
    clock.value += 6
    assert watchdog.step() == WatchdogDecision("wait")
    assert inspector.terminated == [orphan.pid]
    assert tmux.launched == []
    assert any(event.get("reason") ==
               "tracked-leader-exited-before-session"
               for event in logger.events)
    assert watchdog.step() == WatchdogDecision("wait")
    assert watchdog.tracked is not None
    assert len(tmux.launched) == 1


def test_unrecognized_finalizer_descendant_is_cleaned_before_restart(
        tmp_path: Path) -> None:
    watchdog, inspector, tmux, _, clock, _ = _watchdog(
        tmp_path, orphan_grace_seconds=5)
    _make_tracked_process_disappear(watchdog, inspector, tmux)
    helper = ProcessIdentity(
        pid=777, uid=os.geteuid(), ppid=100, pgrp=777, session=100,
        start_ticks=7777, state="S", cwd=watchdog.repo_root,
        argv=(str(watchdog.repo_root / ".venv/bin/python"),
              "scripts/sage_mem_v1_formal_finalizer.py", "--execute"))
    inspector.processes[helper.pid] = helper
    assert watchdog.step() == WatchdogDecision("wait")
    assert tmux.launched == []
    clock.value += 6
    assert watchdog.step() == WatchdogDecision("wait")
    assert helper.pid in inspector.terminated
    assert tmux.launched == []
    assert watchdog.step() == WatchdogDecision("wait")
    assert len(tmux.launched) == 1


def test_restart_counter_resets_only_after_healthy_new_progress(
        tmp_path: Path) -> None:
    watchdog, inspector, tmux, _, clock, progress = _watchdog(
        tmp_path, healthy_reset_seconds=10)
    _make_tracked_process_disappear(watchdog, inspector, tmux)
    watchdog.step()
    assert watchdog.restart_attempts == 1
    progress.append((11, 0))
    clock.value += 9
    watchdog.step()
    assert watchdog.restart_attempts == 1
    clock.value += 2
    watchdog.step()
    assert watchdog.restart_attempts == 0


def test_tmux_launch_must_resolve_to_exact_supervisor_identity(
        tmp_path: Path) -> None:
    watchdog, inspector, tmux, logger, _, _ = _watchdog(
        tmp_path, launch_identity_valid=False)
    _make_tracked_process_disappear(watchdog, inspector, tmux)
    assert watchdog.step() == WatchdogDecision("wait")
    assert watchdog.tracked is None
    assert len(tmux.launched) == 1
    assert tmux.launched[0] in tmux.killed
    assert any(event.get("reason") ==
               "exact-process-verification-timeout"
               for event in logger.events)


def test_signal_during_restart_aborts_and_terminates_new_session(
        tmp_path: Path) -> None:
    watchdog, inspector, tmux, logger, _, _ = _watchdog(tmp_path)
    _make_tracked_process_disappear(watchdog, inspector, tmux)
    original_launch = tmux.launch

    def launch_and_signal(*args: object, **kwargs: object) -> bool:
        launched = original_launch(*args, **kwargs)
        watchdog.request_stop()
        return launched

    tmux.launch = launch_and_signal  # type: ignore[method-assign]
    assert watchdog.step() == WatchdogDecision("exit", 0)
    assert watchdog.tracked is None
    assert inspector.processes == {}
    assert len(tmux.launched) == 1
    assert tmux.launched[0] in tmux.killed
    assert any(event["event"] == "restart-aborted"
               and event.get("terminated_and_verified") is True
               for event in logger.events)


def test_operational_error_during_restart_terminates_launch_candidate(
        tmp_path: Path) -> None:
    watchdog, inspector, tmux, _, _, _ = _watchdog(tmp_path)
    _make_tracked_process_disappear(watchdog, inspector, tmux)

    def fail_progress(_path: Path) -> tuple[int, int]:
        raise OSError("progress filesystem failed")

    watchdog.progress_reader = fail_progress
    assert watchdog.run() == 4
    assert inspector.processes == {}
    assert len(tmux.launched) == 1
    assert tmux.launched[0] in tmux.killed


def test_gpu_temperature_warning_is_noninvasive_by_default(
        tmp_path: Path) -> None:
    watchdog, _, tmux, logger, _, _ = _watchdog(
        tmp_path, temperatures={0: 75, 1: 91, 2: 88})
    assert watchdog.step() == WatchdogDecision("wait")
    assert watchdog.step() == WatchdogDecision("wait")
    warnings = [event for event in logger.events
                if event["event"] == "gpu-temperature-warning"]
    assert len(warnings) == 1
    assert warnings[0]["physical_gpu"] == 1
    assert warnings[0]["fail_closed_enabled"] is False
    assert tmux.killed == []


def test_explicit_gpu_stop_threshold_is_fail_closed(tmp_path: Path) -> None:
    watchdog, _, tmux, logger, _, _ = _watchdog(
        tmp_path, temperatures={0: 75, 1: 93, 2: 88},
        gpu_stop_celsius=92)
    assert watchdog.step() == WatchdogDecision("exit", 2)
    assert "initial" in tmux.killed
    assert any(event.get("reason") == "gpu-temperature-threshold"
               for event in logger.events)


def test_configured_gpu_stop_probe_failure_is_fail_closed(
        tmp_path: Path) -> None:
    watchdog, inspector, tmux, logger, _, _ = _watchdog(
        tmp_path, gpu_stop_celsius=92)

    def fail_probe() -> dict[int, int]:
        raise WatchdogError("probe unavailable")

    watchdog.gpu_reader = fail_probe
    assert watchdog.step() == WatchdogDecision("exit", 2)
    assert inspector.processes == {}
    assert tmux.killed == ["initial"]
    assert any(event.get("reason") == "gpu-temperature-probe-unavailable"
               for event in logger.events)


def test_operational_error_terminates_campaign_before_watchdog_exits(
        tmp_path: Path) -> None:
    watchdog, inspector, _, _, _, _ = _watchdog(tmp_path)

    def fail_resources(_path: Path) -> ResourceSnapshot:
        raise OSError("resource filesystem disappeared")

    watchdog.resource_reader = fail_resources
    assert watchdog.run() == 4
    assert inspector.processes == {}


def test_event_log_failure_terminates_campaign_before_watchdog_exits(
        tmp_path: Path) -> None:
    watchdog, inspector, _, _, _, progress = _watchdog(tmp_path)

    class FailingLogger:
        def emit(self, _event: str, **_fields: object) -> None:
            raise OSError("event log is unwritable")

    watchdog.logger = FailingLogger()
    progress.append((11, 0))
    assert watchdog.run() == 4
    assert inspector.processes == {}


def test_valid_completion_exits_without_resource_or_process_mutation(
        tmp_path: Path) -> None:
    watchdog, _, tmux, _, _, _ = _watchdog(tmp_path)
    _write_completion_bundle(watchdog.study_root)
    assert watchdog.step() == WatchdogDecision("exit", 0)
    assert tmux.killed == []
    assert tmux.launched == []


def test_invalid_completion_report_stops_tracked_campaign(
        tmp_path: Path) -> None:
    watchdog, _, tmux, logger, _, _ = _watchdog(tmp_path)
    _write_completion_bundle(watchdog.study_root)
    value = _valid_report()
    value["finalized_cells_verified"] = 599
    watchdog.report_path.write_text(json.dumps(value), encoding="utf-8")
    assert watchdog.step() == WatchdogDecision("exit", 4)
    assert tmux.killed == ["initial"]
    assert any(event.get("reason") ==
               "report-finalized_cells_verified-mismatch"
               for event in logger.events)


def test_tmux_identity_drift_terminates_exact_process_group(
        tmp_path: Path) -> None:
    watchdog, inspector, tmux, _, _, _ = _watchdog(tmp_path)
    tmux.panes["initial"] = PaneIdentity(999, False, None)
    assert watchdog.step() == WatchdogDecision("exit", 4)
    assert inspector.terminated == [100]


def test_jsonl_event_log_is_append_only_and_parseable(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log = EventLog(path)
    log.emit("first", value=1)
    log.emit("second", value=2)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["event"] for record in records] == ["first", "second"]
    assert all("timestamp_utc" in record for record in records)


def test_watchdog_lock_prevents_two_restart_state_machines(
        tmp_path: Path) -> None:
    path = tmp_path / "watchdog.lock"
    first = acquire_watchdog_lock(path)
    try:
        with pytest.raises(WatchdogError, match="another campaign watchdog"):
            acquire_watchdog_lock(path)
    finally:
        os.close(first)
    replacement = acquire_watchdog_lock(path)
    os.close(replacement)
