#!/usr/bin/env python3
"""Build the fail-closed V18 restart/provenance audit.

This generator deliberately does not parse or report scientific metrics.  It
checks terminal runner/analyzer receipts, hashes artifact bytes, verifies the
five known interrupted/replacement lineages (including remote W&B terminal
state), and writes one schema-v2 JSON document atomically.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 2
EXPECTED_CELLS = 200
EXPECTED_EPOCHS = 100
EXPECTED_COMMANDS_SHA256 = (
    "a8d90d2374ecd2d6eb3d76278e98f9ade31a98d966b0f90a1f1351ed5a0bf7a0"
)
EXPECTED_PROTOCOL_SHA256 = (
    "357cbe12969268020c1a3b96d14542760ebce0ce600fceb6193aaa3c87fe74b2"
)
EXPECTED_STALE_AUDIT_SHA256 = (
    "f3797fe1069775529b98f37dea82a9e0124069d8d53a93776500af424e0918ce"
)
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()

OUTPUT_ROOT_REL = Path("outputs/lewm_v8_v18_confirmation")
LOG_ROOT_REL = Path("logs/lewm_v8_v18_confirmation")
PROTOCOL_REL = OUTPUT_ROOT_REL / "confirmation_protocol.json"
RUNS_REL = OUTPUT_ROOT_REL / "confirmation_runs.json"
ATTEMPTS_REL = OUTPUT_ROOT_REL / "confirmation_attempts.json"
SUMMARY_REL = OUTPUT_ROOT_REL / "confirmation_summary.json"
ANALYSIS_REL = OUTPUT_ROOT_REL / "confirmation_analysis.json"
CELLS_CSV_REL = OUTPUT_ROOT_REL / "confirmation_cells.csv"
CONTRASTS_CSV_REL = OUTPUT_ROOT_REL / "confirmation_contrasts.csv"
LOCK_REL = OUTPUT_ROOT_REL / ".lewm_v8_v18_confirmation.lock"
STALE_AUDIT_REL = Path(".paper-draft/v18_restart_audit.json")
ORIGINAL_RUNNER_LOG_REL = Path("logs/lewm_v8_v18_confirmation_runner.log")
FIRST_RESUME_LOG_REL = LOG_ROOT_REL / "resume_driver.log"

CORE_ARTIFACTS = (
    "model.pt",
    "metrics.json",
    "eval_rollout.npz",
    "wandb_run.json",
)
TASKS = (
    "acrobot.swingup",
    "manipulator.bring_ball",
    "quadruped.run",
    "stacker.stack_4",
    "swimmer.swimmer15",
)
DESIGNS = (
    "vicreg_none",
    "vicreg_gru",
    "vicreg_ssm",
    "vicreg_hacssmv8",
    "vicreg_hacssmv8_static",
    "vicreg_hacssmv8_dynamic",
    "vicreg_hacssmv8_noaction",
    "vicreg_hacssmv8_single",
)
SEEDS = (18001, 18002, 18003, 18004, 18005)
EXPECTED_CELL_SET = {
    (task, design, seed)
    for task in TASKS
    for seed in SEEDS
    for design in DESIGNS
}
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
EPOCH_RE = re.compile(r"^e\s*(\d+)/(\d+)\b")
ACTIVE_SCRIPT_NAMES = {
    "run_lewm_v8_v18.py",
    "train_lewm_v8_v18.py",
    "analyze_lewm_v8_v18.py",
}


def _lineage(
    *,
    task: str,
    design: str,
    seed: int,
    command_sha256: str,
    interrupted_log: str,
    interrupted_log_sha256: str,
    interrupted_epoch: int,
    interrupted_wandb_id: str,
    interrupted_wandb_run_dir: str,
    interrupted_wandb_file_sha256: str,
    interrupted_created_at: str,
    interrupted_heartbeat_at: str,
    interrupted_history_lines: int,
    replacement_log: str,
    replacement_log_sha256: str,
    replacement_wandb_id: str,
    replacement_wandb_run_dir: str,
    replacement_wandb_file_sha256: str,
    replacement_created_at: str,
    replacement_heartbeat_at: str,
    replacement_completed_at: str,
    artifact_sha256: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "cell": {"task": task, "design": design, "seed": seed},
        "command_sha256": command_sha256,
        "interrupted": {
            "log": interrupted_log,
            "log_sha256": interrupted_log_sha256,
            "last_logged_epoch": interrupted_epoch,
            "wandb_id": interrupted_wandb_id,
            "wandb_state": "crashed",
            "wandb_run_dir": interrupted_wandb_run_dir,
            "wandb_file_sha256": interrupted_wandb_file_sha256,
            "wandb_created_at": interrupted_created_at,
            "wandb_heartbeat_at": interrupted_heartbeat_at,
            "wandb_history_lines": interrupted_history_lines,
        },
        "replacement": {
            "log": replacement_log,
            "log_sha256": replacement_log_sha256,
            "first_logged_epoch": 1,
            "terminal_logged_epoch": EXPECTED_EPOCHS,
            "wandb_id": replacement_wandb_id,
            "wandb_state": "finished",
            "wandb_run_dir": replacement_wandb_run_dir,
            "wandb_file_sha256": replacement_wandb_file_sha256,
            "wandb_created_at": replacement_created_at,
            "wandb_heartbeat_at": replacement_heartbeat_at,
            "wandb_history_lines": EXPECTED_EPOCHS,
            "ledger_completed_at": replacement_completed_at,
            "artifact_sha256": dict(artifact_sha256),
        },
    }


LINEAGES: tuple[dict[str, Any], ...] = (
    _lineage(
        task="acrobot.swingup",
        design="vicreg_ssm",
        seed=18005,
        command_sha256="32ae51a7ef73a63e81a1daaaf460cd9864d82d9b06441210a6e1dddb79370b2e",
        interrupted_log="logs/lewm_v8_v18_confirmation/dmc_acrobot_swingup-vicreg_ssm-s18005.log",
        interrupted_log_sha256="dfe369fac5d9c40a7a8e561d7ed94bc0ca4f8d4f22459d911e6fb46e01e7198e",
        interrupted_epoch=62,
        interrupted_wandb_id="s2zw6r98",
        interrupted_wandb_run_dir=(
            "outputs/lewm_v8_v18_confirmation/"
            "lewm-dmc:acrobot.swingup-vicreg_ssm-s18005/"
            "wandb/run-20260702_073109-s2zw6r98"
        ),
        interrupted_wandb_file_sha256="68a6088808c2cf7d7b01e4a2bc839e0df59ee37fa515fe32baac9eacd5b61884",
        interrupted_created_at="2026-07-01T23:31:10Z",
        interrupted_heartbeat_at="2026-07-01T23:36:41Z",
        interrupted_history_lines=60,
        replacement_log="logs/lewm_v8_v18_confirmation/dmc_acrobot_swingup-vicreg_ssm-s18005.attempt2.log",
        replacement_log_sha256="de69a7091d8aae9eb268ef154e23c50dc675c9ffc2834063a9819d7e385bfd46",
        replacement_wandb_id="purj7jwl",
        replacement_wandb_run_dir=(
            "outputs/lewm_v8_v18_confirmation/"
            "lewm-dmc:acrobot.swingup-vicreg_ssm-s18005/"
            "wandb/run-20260702_090630-purj7jwl"
        ),
        replacement_wandb_file_sha256="1e36299eba69dbb274cfe135b60cab7f57128a6fa62f4d466faf5a17f0fbd4dd",
        replacement_created_at="2026-07-02T01:06:31Z",
        replacement_heartbeat_at="2026-07-02T01:16:12Z",
        replacement_completed_at="2026-07-02T01:16:14.188151+00:00",
        artifact_sha256={
            "model.pt": "7dc63e83f71459b14bc4441910e15d1fb636c512ade8d38c4e962110c92b02f6",
            "metrics.json": "92d5134b5a2a0606343e51ab3901df5acd3ce3b6822576c3c5ee1c833442ef34",
            "eval_rollout.npz": "77179b94f096f31f6028302d5f39339b980b508db58e7118c22402525bf36637",
            "wandb_run.json": "c1201ef9f68432c133b443a25281cdebcd6cc3b0e008cde50e8c921c19a89ef6",
        },
    ),
    _lineage(
        task="manipulator.bring_ball",
        design="vicreg_ssm",
        seed=18005,
        command_sha256="11f40b5321dfb29f47b4384f80914e47f6dd6b3a8f101887afb6ee0fc22f429f",
        interrupted_log="logs/lewm_v8_v18_confirmation/dmc_manipulator_bring_ball-vicreg_ssm-s18005.log",
        interrupted_log_sha256="3492682adbf9d3f75e4cdf2df9c8e3ccb56ddea65212b9408bd5c89b17cc722e",
        interrupted_epoch=31,
        interrupted_wandb_id="lvnu86iv",
        interrupted_wandb_run_dir=(
            "outputs/lewm_v8_v18_confirmation/"
            "lewm-dmc:manipulator.bring_ball-vicreg_ssm-s18005/"
            "wandb/run-20260702_073358-lvnu86iv"
        ),
        interrupted_wandb_file_sha256="af9d26661a0cf1c733b91bfe8de8227c473b7fba3cc65b718d6cd208ea0255ed",
        interrupted_created_at="2026-07-01T23:33:58Z",
        interrupted_heartbeat_at="2026-07-01T23:36:44Z",
        interrupted_history_lines=29,
        replacement_log="logs/lewm_v8_v18_confirmation/dmc_manipulator_bring_ball-vicreg_ssm-s18005.attempt2.log",
        replacement_log_sha256="2c516e0b6f8cea92b9de93ff5ef18a20fb913c41ff9c4094f9bd6c24bf199956",
        replacement_wandb_id="jj02okok",
        replacement_wandb_run_dir=(
            "outputs/lewm_v8_v18_confirmation/"
            "lewm-dmc:manipulator.bring_ball-vicreg_ssm-s18005/"
            "wandb/run-20260702_090632-jj02okok"
        ),
        replacement_wandb_file_sha256="4db95f9255031c3756d0eb9fbda1bd9f2b96cccf0789170006213c412361a99d",
        replacement_created_at="2026-07-02T01:06:33Z",
        replacement_heartbeat_at="2026-07-02T01:16:12Z",
        replacement_completed_at="2026-07-02T01:16:14.566177+00:00",
        artifact_sha256={
            "model.pt": "7d951df9624dae59241dcb4814a87b00c5eed3911459aea11b56e2c3f7318e3a",
            "metrics.json": "11f6638f4f325645f7b57fef5c912065e16aec69dbc9f6482e36096aaca0fbfb",
            "eval_rollout.npz": "c8903960e07ac845ccb24de2735e3cce87fe11e34f419d09424f1895d4433ed7",
            "wandb_run.json": "98f0a90d2e4419c98847eda3d1a6d4eef5941877d718936717e391bf150690a9",
        },
    ),
    _lineage(
        task="quadruped.run",
        design="vicreg_ssm",
        seed=18005,
        command_sha256="33a04ae52958affbbd21a8b59bef87ca81976a15e611cfa7d9bcaa641cb919cf",
        interrupted_log="logs/lewm_v8_v18_confirmation/dmc_quadruped_run-vicreg_ssm-s18005.log",
        interrupted_log_sha256="6a72339d3e8e81fe9a60c82a641602440dbece11559c4229a03b731f4f1971ed",
        interrupted_epoch=9,
        interrupted_wandb_id="67cg0ir3",
        interrupted_wandb_run_dir=(
            "outputs/lewm_v8_v18_confirmation/"
            "lewm-dmc:quadruped.run-vicreg_ssm-s18005/"
            "wandb/run-20260702_073556-67cg0ir3"
        ),
        interrupted_wandb_file_sha256="ab8879c80aaf6146cf198cc2fc3178d1fa598c892078b6d7ef2bcfdb231f1f8d",
        interrupted_created_at="2026-07-01T23:35:56Z",
        interrupted_heartbeat_at="2026-07-01T23:36:42Z",
        interrupted_history_lines=7,
        replacement_log="logs/lewm_v8_v18_confirmation/dmc_quadruped_run-vicreg_ssm-s18005.attempt2.log",
        replacement_log_sha256="6f005ea3ae5157a958deefa02c9a727d1885d9fcf6e6c540dcd09b95062b507b",
        replacement_wandb_id="p4yhjd8a",
        replacement_wandb_run_dir=(
            "outputs/lewm_v8_v18_confirmation/"
            "lewm-dmc:quadruped.run-vicreg_ssm-s18005/"
            "wandb/run-20260702_090644-p4yhjd8a"
        ),
        replacement_wandb_file_sha256="22681cac82484a988aa6258326d5bf19634d71dac3959de8aa295738945fe52f",
        replacement_created_at="2026-07-02T01:06:45Z",
        replacement_heartbeat_at="2026-07-02T01:16:19Z",
        replacement_completed_at="2026-07-02T01:16:21.477129+00:00",
        artifact_sha256={
            "model.pt": "aca0efda6f4d9ac5c18d03cd189145de3f57efac005f66a5663f88d51b6734e0",
            "metrics.json": "198ffdfde0afbef232c5eb4a6fd5485a6763e4a37dd2757c78e4bd86378d215d",
            "eval_rollout.npz": "0def0c11ff360b75489fcb447bb2d0a50344ee222e98cfbb331939408871d527",
            "wandb_run.json": "5c24e64996e2b7e0687152e9b603fb2baf125afdb6e6e327592363c18f7e7f7f",
        },
    ),
    _lineage(
        task="swimmer.swimmer15",
        design="vicreg_ssm",
        seed=18005,
        command_sha256="c2b718d90745cf4bc5107b45eb04ed6e37f5ec69411dddabef308a34ff2937d9",
        interrupted_log="logs/lewm_v8_v18_confirmation/dmc_swimmer_swimmer15-vicreg_ssm-s18005.log",
        interrupted_log_sha256="4029c6a1b1be00bf03742dd624993b247c1d9f328a39bbbdc67456e01fd1cc9f",
        interrupted_epoch=62,
        interrupted_wandb_id="1txbup0i",
        interrupted_wandb_run_dir=(
            "outputs/lewm_v8_v18_confirmation/"
            "lewm-dmc:swimmer.swimmer15-vicreg_ssm-s18005/"
            "wandb/run-20260702_073110-1txbup0i"
        ),
        interrupted_wandb_file_sha256="c0523df33b0b7ba3ba344089d78bb2202b9a42dc981cd07ed8ea5f724befb819",
        interrupted_created_at="2026-07-01T23:31:11Z",
        interrupted_heartbeat_at="2026-07-01T23:36:42Z",
        interrupted_history_lines=60,
        replacement_log="logs/lewm_v8_v18_confirmation/dmc_swimmer_swimmer15-vicreg_ssm-s18005.attempt2.log",
        replacement_log_sha256="5e081524231378c5e9c2d75887f2abb6a41670bf4aa03b09da1f5530a49e3147",
        replacement_wandb_id="5yeoueiz",
        replacement_wandb_run_dir=(
            "outputs/lewm_v8_v18_confirmation/"
            "lewm-dmc:swimmer.swimmer15-vicreg_ssm-s18005/"
            "wandb/run-20260702_090634-5yeoueiz"
        ),
        replacement_wandb_file_sha256="439dbd7559df2f980216a7fda8e712014db3b10c6e6cf2e91997073c7aa7d061",
        replacement_created_at="2026-07-02T01:06:35Z",
        replacement_heartbeat_at="2026-07-02T01:16:09Z",
        replacement_completed_at="2026-07-02T01:16:11.615656+00:00",
        artifact_sha256={
            "model.pt": "66c6c1ba58d09855c9a94c7be2e174e35cbdecd2ac89537864744535016f9f5b",
            "metrics.json": "040af12c2e7f8b2adbcba23b8943581e8c6e40cc573860cb96d1e777d5e4ca24",
            "eval_rollout.npz": "a8153128c65d1916d94147feed7d7447cc02a786cdd7cb469bdbe01948dc4d33",
            "wandb_run.json": "acbd0099bd9817327f1deedd05161a6cceee9356792204319dd3b0c53df2e38a",
        },
    ),
    _lineage(
        task="stacker.stack_4",
        design="vicreg_hacssmv8_static",
        seed=18003,
        command_sha256="2fc644ee408b1203c0684a1f40fb0b084fc6351209ac08b51b0d51b85e963556",
        interrupted_log="logs/lewm_v8_v18_confirmation/dmc_stacker_stack_4-vicreg_hacssmv8_static-s18003.log",
        interrupted_log_sha256="e74f6ba0dcb1e07514005008f8a02e0a707de99f0b812d61875a79de5f435050",
        interrupted_epoch=37,
        interrupted_wandb_id="6e455hnn",
        interrupted_wandb_run_dir=(
            "outputs/lewm_v8_v18_confirmation/"
            "lewm-dmc:stacker.stack_4-vicreg_hacssmv8_static-s18003/"
            "wandb/run-20260702_134910-6e455hnn"
        ),
        interrupted_wandb_file_sha256="08cc7b344f8fe1ab544374e030230f1011a0659ef2dc7a1e23004e631baa6765",
        interrupted_created_at="2026-07-02T05:49:11Z",
        interrupted_heartbeat_at="2026-07-02T05:52:56Z",
        interrupted_history_lines=34,
        replacement_log="logs/lewm_v8_v18_confirmation/dmc_stacker_stack_4-vicreg_hacssmv8_static-s18003.attempt2.log",
        replacement_log_sha256="63b9ee95f0ebe656b2bccaa1048cc11c5f0440e9ebb4d1962265acf890ebcb76",
        replacement_wandb_id="vclq4lns",
        replacement_wandb_run_dir=(
            "outputs/lewm_v8_v18_confirmation/"
            "lewm-dmc:stacker.stack_4-vicreg_hacssmv8_static-s18003/"
            "wandb/run-20260702_230545-vclq4lns"
        ),
        replacement_wandb_file_sha256="590e35c24536dedd6171b45e1361c0de68e418d9d49cf1815dd09a6f49135fdb",
        replacement_created_at="2026-07-02T15:05:46Z",
        replacement_heartbeat_at="2026-07-02T15:23:05Z",
        replacement_completed_at="2026-07-02T15:29:31.280515+00:00",
        artifact_sha256={
            "model.pt": "cf8b4905803bdb350247821a90770969d3a31bf3ea8f47e45895bf897bdf745f",
            "metrics.json": "94eacf99b8ddd039355a0bfecfb96de0e3790248226ea5093fd479782f980c26",
            "eval_rollout.npz": "3196bc8498e1c85b6568e07fa18a51b295ab7fe1a00208dfdb15ab3f28880888",
            "wandb_run.json": "5c676dfe1e6579a83a6c9808aea35c60c35c83b8d1dc9d4a509611447308c17f",
        },
    ),
)

PARENT_ONLY_PAUSE: dict[str, Any] = {
    "event": "runner_parent_resource_pause",
    "evidence_class": "operator_transcript",
    "parent_pid": 3171896,
    "trainer_pid": 3173946,
    "cell": {
        "task": "stacker.stack_4",
        "design": "vicreg_hacssmv8_static",
        "seed": 18003,
    },
    "sigstop_at_approximately": "2026-07-02T15:19:13Z",
    "stopped_state_observed_at": "2026-07-02T15:19:55Z",
    "trainer_signaled": False,
    "trainer_continued": True,
    "trainer_exited_cleanly_at": "2026-07-02T15:23:16Z",
    "gpu_competitor_released_at": "2026-07-02T15:29:08Z",
    "sigcont_at_approximately": "2026-07-02T15:29:29Z",
    "parent_committed_completed_child_at": "2026-07-02T15:29:31.280515+00:00",
    "classification": "runner_parent_pause_only",
    "trainer_interrupted": False,
    "restart_required": False,
    "cell_count_effect": 0,
}


class AuditError(RuntimeError):
    """A required terminal or provenance condition is not satisfied."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot load JSON {path}: {exc}") from exc


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def parse_utc(value: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise AuditError(f"invalid UTC timestamp {value!r}") from exc
    require(parsed.tzinfo is not None, f"timestamp lacks timezone: {value!r}")
    return parsed.astimezone(dt.timezone.utc)


def ns_utc(value: int) -> str:
    seconds, nanos = divmod(value, 1_000_000_000)
    base = dt.datetime.fromtimestamp(seconds, tz=dt.timezone.utc)
    return f"{base:%Y-%m-%dT%H:%M:%S}.{nanos:09d}Z"


def relative_display(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def file_receipt(path: Path, root: Path) -> dict[str, Any]:
    require(path.is_file(), f"required file is absent: {path}")
    stat = path.stat()
    return {
        "path": relative_display(path, root),
        "size_bytes": stat.st_size,
        "mtime_utc": ns_utc(stat.st_mtime_ns),
        "sha256": file_sha256(path),
    }


def assert_hash(path: Path, expected: str) -> dict[str, Any]:
    require(HEX64_RE.fullmatch(expected) is not None, f"bad expected hash: {expected}")
    actual = file_sha256(path)
    require(actual == expected, f"SHA-256 mismatch for {path}: {actual} != {expected}")
    return {"path": str(path), "sha256": actual}


def cell_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    try:
        return str(row["task"]), str(row["design"]), int(row["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AuditError(f"malformed cell row identity: {row!r}") from exc


def validate_epoch_log(path: Path, *, terminal_epoch: int, completed: bool) -> dict[str, Any]:
    epochs: list[int] = []
    totals: list[int] = []
    saw_traceback = False
    saw_sync_finish = False
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        for line in stream:
            match = EPOCH_RE.match(line)
            if match:
                epochs.append(int(match.group(1)))
                totals.append(int(match.group(2)))
            if line.startswith("Traceback (most recent call last):"):
                saw_traceback = True
            if line.startswith("wandb: Synced "):
                saw_sync_finish = True
    require(epochs, f"no epoch receipts in {path}")
    require(epochs == list(range(1, terminal_epoch + 1)), f"epoch sequence differs in {path}")
    require(set(totals) == {EXPECTED_EPOCHS}, f"epoch denominator differs in {path}")
    require(not saw_traceback, f"unexpected traceback marker in {path}")
    require(saw_sync_finish is completed, f"W&B finish marker differs in {path}")
    return {
        "first_logged_epoch": epochs[0],
        "last_logged_epoch": epochs[-1],
        "planned_epochs": EXPECTED_EPOCHS,
        "traceback_marker_present": saw_traceback,
        "normal_wandb_finish_marker_present": saw_sync_finish,
    }


def active_v18_processes() -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    proc = Path("/proc")
    if not proc.is_dir():
        raise AuditError("/proc is required to prove final V18 runner exit")
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        argv = [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]
        script_names = {Path(arg).name for arg in argv}
        matched = sorted(script_names & ACTIVE_SCRIPT_NAMES)
        if matched:
            found.append({"pid": int(entry.name), "scripts": matched, "argv": argv})
    return sorted(found, key=lambda row: row["pid"])


def assert_terminal_quiescence(root: Path) -> None:
    lock = root / LOCK_REL
    require(not lock.exists(), f"V18 lock still exists: {lock}")
    active = active_v18_processes()
    require(not active, f"V18 runner/trainer/analyzer process still active: {active}")


def validate_protocol(protocol: Any, protocol_path: Path) -> tuple[dict[tuple[str, str, int], list[str]], dict[str, Any]]:
    require(isinstance(protocol, Mapping), "protocol must be an object")
    require(file_sha256(protocol_path) == EXPECTED_PROTOCOL_SHA256, "frozen protocol file hash differs")
    require(protocol.get("schema_version") == 1, "protocol schema differs")
    require(protocol.get("scope") == "lewm_v8_v18_unopened_task_confirmation", "protocol scope differs")
    require(protocol.get("study") == "lewm-v8-v18-confirmation", "protocol study differs")
    require(protocol.get("runs") == EXPECTED_CELLS, "protocol run count differs")
    require(protocol.get("epochs") == EXPECTED_EPOCHS, "protocol epochs differ")
    require(tuple(protocol.get("tasks", ())) == TASKS, "protocol task tuple differs")
    require(tuple(protocol.get("designs", ())) == DESIGNS, "protocol design tuple differs")
    require(tuple(protocol.get("seeds", ())) == SEEDS, "protocol seed tuple differs")
    require(protocol.get("resume_supported") is True, "protocol resume support differs")
    require(protocol.get("resume_granularity") == "complete_cell_only", "protocol resume policy differs")
    require(protocol.get("wandb_enabled") is True, "protocol W&B requirement differs")
    require(protocol.get("wandb_mode") == "online", "protocol W&B mode differs")
    commands = protocol.get("commands")
    require(isinstance(commands, list) and len(commands) == EXPECTED_CELLS, "protocol commands differ")
    require(json_sha256(commands) == EXPECTED_COMMANDS_SHA256, "protocol command-list hash differs")
    require(protocol.get("commands_sha256") == EXPECTED_COMMANDS_SHA256, "recorded command hash differs")
    indexed: dict[tuple[str, str, int], list[str]] = {}
    for row in commands:
        require(isinstance(row, Mapping), "protocol command row is not an object")
        key = cell_key(row)
        argv = row.get("argv")
        require(isinstance(argv, list) and all(isinstance(value, str) for value in argv), f"bad argv for {key}")
        require(key not in indexed, f"duplicate protocol command {key}")
        indexed[key] = list(argv)
    require(set(indexed) == EXPECTED_CELL_SET, "protocol command cell set differs")
    receipt = {
        "path": PROTOCOL_REL.as_posix(),
        "sha256": EXPECTED_PROTOCOL_SHA256,
        "commands_sha256": EXPECTED_COMMANDS_SHA256,
        "created_at": protocol.get("created_at"),
        "planned_cells": EXPECTED_CELLS,
        "epochs_per_cell": EXPECTED_EPOCHS,
        "resume_policy": "complete_cell_only",
    }
    return indexed, receipt


def validate_summary(summary: Any) -> dict[str, Any]:
    require(isinstance(summary, Mapping), "summary must be an object")
    require(summary.get("schema_version") == 1, "summary schema differs")
    require(summary.get("scope") == "lewm_v8_v18_unopened_task_confirmation", "summary scope differs")
    require(summary.get("status") == "COMPLETE", "V18 is not COMPLETE")
    require(summary.get("expected_cells") == EXPECTED_CELLS, "summary expected count differs")
    require(summary.get("completed_cells") == EXPECTED_CELLS, "refusing before COMPLETE 200/200")
    require(summary.get("failed_or_invalid_cells") == 0, "summary has failed/invalid cells")
    require(summary.get("failures") == [], "summary failure list is nonempty")
    require(summary.get("wandb_enabled") is True, "summary W&B flag differs")
    require(summary.get("resume") is True, "final summary is not the verified resume completion")
    finished_at = summary.get("finished_at")
    require(isinstance(finished_at, str), "summary finish time is missing")
    parse_utc(finished_at)
    return {
        "status": "COMPLETE",
        "expected_cells": EXPECTED_CELLS,
        "completed_cells": EXPECTED_CELLS,
        "failed_or_invalid_cells": 0,
        "finished_at": finished_at,
        "resume": True,
    }


def validate_ledgers(
    root: Path,
    commands: Mapping[tuple[str, str, int], Sequence[str]],
    runs: Any,
    attempts: Any,
) -> tuple[
    dict[tuple[str, str, int], Mapping[str, Any]],
    dict[tuple[str, str, int], Mapping[str, Any]],
    dict[str, Any],
]:
    require(isinstance(runs, list) and len(runs) == EXPECTED_CELLS, "run ledger must have 200 rows")
    require(isinstance(attempts, list) and len(attempts) == EXPECTED_CELLS, "attempt ledger must have 200 rows")

    def index(rows: list[Any], label: str) -> dict[tuple[str, str, int], Mapping[str, Any]]:
        result: dict[tuple[str, str, int], Mapping[str, Any]] = {}
        for raw in rows:
            require(isinstance(raw, Mapping), f"{label} contains a non-object")
            key = cell_key(raw)
            require(key not in result, f"{label} contains duplicate {key}")
            require(raw.get("status") == "complete", f"{label} has noncomplete {key}")
            require(raw.get("wandb_state") == "finished", f"{label} has nonfinished W&B {key}")
            completed_at = raw.get("completed_at")
            require(isinstance(completed_at, str), f"{label} completion time missing for {key}")
            parse_utc(completed_at)
            require(raw.get("command_sha256") == json_sha256(commands.get(key)), f"{label} command hash differs for {key}")
            hashes = raw.get("artifact_sha256")
            require(isinstance(hashes, Mapping) and set(hashes) == set(CORE_ARTIFACTS), f"{label} artifact map differs for {key}")
            require(all(isinstance(value, str) and HEX64_RE.fullmatch(value) for value in hashes.values()), f"{label} has malformed artifact hash for {key}")
            result[key] = raw
        require(set(result) == EXPECTED_CELL_SET, f"{label} cell set differs")
        return result

    run_index = index(runs, "run ledger")
    attempt_index = index(attempts, "attempt ledger")
    resumed_existing = sum(row.get("resumed_existing") is True for row in runs)
    executed_on_final_resume = sum(row.get("resumed_existing") is False for row in runs)
    require((resumed_existing, executed_on_final_resume) == (180, 20), "final run-ledger resume split differs")
    require(all(row.get("resumed_existing") is False for row in attempts), "attempt ledger contains a resumed-existing pseudo-attempt")
    local_wandb_ids: set[str] = set()
    checked_files = 0
    checked_bytes = 0
    for key in sorted(EXPECTED_CELL_SET):
        run = run_index[key]
        attempt = attempt_index[key]
        require(run.get("artifact_sha256") == attempt.get("artifact_sha256"), f"ledger artifact maps differ for {key}")
        require(run.get("command_sha256") == attempt.get("command_sha256"), f"ledger command hashes differ for {key}")
        directory_raw = run.get("directory")
        require(isinstance(directory_raw, str) and directory_raw, f"run directory missing for {key}")
        directory = Path(directory_raw).resolve()
        try:
            require(directory.parent == (root / OUTPUT_ROOT_REL).resolve(), f"run directory escaped output root for {key}")
        except OSError as exc:
            raise AuditError(f"cannot resolve run directory for {key}: {exc}") from exc
        hashes = run["artifact_sha256"]
        for name in CORE_ARTIFACTS:
            path = directory / name
            require(path.is_file() and path.stat().st_size > 0, f"missing core artifact {path}")
            actual = file_sha256(path)
            require(actual == hashes[name], f"core artifact hash differs: {path}")
            checked_files += 1
            checked_bytes += path.stat().st_size
        receipt = load_json(directory / "wandb_run.json")
        require(isinstance(receipt, Mapping), f"W&B receipt is not an object for {key}")
        require(receipt.get("state") == "finished", f"local W&B receipt is not finished for {key}")
        run_id = receipt.get("run_id")
        require(isinstance(run_id, str) and run_id.strip(), f"local W&B run ID missing for {key}")
        require(run_id not in local_wandb_ids, f"duplicate local W&B run ID {run_id}")
        local_wandb_ids.add(run_id)
    require(len(local_wandb_ids) == EXPECTED_CELLS, "local W&B ID count differs")
    return run_index, attempt_index, {
        "checked_cells": EXPECTED_CELLS,
        "checked_core_artifacts": checked_files,
        "checked_core_artifact_bytes": checked_bytes,
        "finished_local_wandb_receipts": len(local_wandb_ids),
        "unique_local_wandb_run_ids": len(local_wandb_ids),
        "final_resume_validated_existing_cells": resumed_existing,
        "final_resume_executed_cells": executed_on_final_resume,
    }


def artifact_manifest_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    entries = [
        {
            "task": str(row.get("task")),
            "seed": int(row.get("seed")),
            "design": str(row.get("design")),
            "artifact_sha256": row.get("artifact_sha256"),
        }
        for row in rows
    ]
    entries.sort(key=lambda row: (row["task"], row["seed"], row["design"]))
    return json_sha256(entries)


def validate_analysis(root: Path, analysis: Any, protocol_hash: str, artifact_manifest_hash: str) -> dict[str, Any]:
    require(isinstance(analysis, Mapping), "analysis must be an object")
    require(analysis.get("schema_version") == 2, "analysis schema differs")
    require(analysis.get("status") == "COMPLETE", "analysis is not COMPLETE")
    require(analysis.get("expected_cells") == EXPECTED_CELLS, "analysis expected count differs")
    require(analysis.get("completed_valid_cells") == EXPECTED_CELLS, "analysis is not 200/200")
    require(analysis.get("artifact_integrity_passed") is True, "analysis artifact integrity failed")
    require(analysis.get("artifact_integrity_errors") == [], "analysis artifact errors are nonempty")
    require(analysis.get("protocol_contract_errors") == [], "analysis protocol errors are nonempty")
    require(analysis.get("input_protocol_sha256") == protocol_hash, "analysis protocol binding differs")
    require(analysis.get("input_artifact_manifest_sha256") == artifact_manifest_hash, "analysis artifact-manifest binding differs")
    cells_hash = file_sha256(root / CELLS_CSV_REL)
    contrasts_hash = file_sha256(root / CONTRASTS_CSV_REL)
    require(analysis.get("cells_csv_sha256") == cells_hash, "analysis cells CSV binding differs")
    require(analysis.get("contrasts_csv_sha256") == contrasts_hash, "analysis contrasts CSV binding differs")
    return {
        "status": "COMPLETE",
        "expected_cells": EXPECTED_CELLS,
        "completed_valid_cells": EXPECTED_CELLS,
        "artifact_integrity_passed": True,
        "input_protocol_sha256": protocol_hash,
        "input_artifact_manifest_sha256": artifact_manifest_hash,
        "cells_csv_sha256": cells_hash,
        "contrasts_csv_sha256": contrasts_hash,
    }


def verify_stale_audit(root: Path) -> dict[str, Any]:
    path = root / STALE_AUDIT_REL
    require(file_sha256(path) == EXPECTED_STALE_AUDIT_SHA256, "stale audit receipt hash differs")
    value = load_json(path)
    require(isinstance(value, Mapping) and value.get("schema_version") == 1, "stale audit schema differs")
    observation = value.get("interruption_observation")
    resume = value.get("resume")
    require(isinstance(observation, Mapping), "stale audit observation missing")
    require(observation.get("complete_valid_cells") == 136, "stale audit initial complete count differs")
    require(observation.get("absent_cells") == 64, "stale audit initial absent count differs")
    require(observation.get("partial_or_invalid_core_cells") == 0, "stale audit partial count differs")
    require(observation.get("interrupted_trainers") == 4, "stale audit interrupted count differs")
    require(isinstance(resume, Mapping) and resume.get("validated_existing_cells") == 136, "stale audit resume count differs")
    return {
        "path": STALE_AUDIT_REL.as_posix(),
        "sha256": EXPECTED_STALE_AUDIT_SHA256,
        "reason": (
            "schema-v1 records only the first four-process interruption and "
            "omits the later stacker-static lineage and parent-only pause classification"
        ),
    }


def verify_local_lineages(
    root: Path,
    commands: Mapping[tuple[str, str, int], Sequence[str]],
    run_index: Mapping[tuple[str, str, int], Mapping[str, Any]],
    attempt_index: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for expected in LINEAGES:
        cell = expected["cell"]
        key = (cell["task"], cell["design"], cell["seed"])
        require(json_sha256(commands[key]) == expected["command_sha256"], f"known command hash differs for {key}")
        require(run_index[key].get("command_sha256") == expected["command_sha256"], f"known run-ledger command differs for {key}")
        require(attempt_index[key].get("command_sha256") == expected["command_sha256"], f"known attempt command differs for {key}")

        interrupted = expected["interrupted"]
        replacement = expected["replacement"]
        interrupted_log = root / interrupted["log"]
        replacement_log = root / replacement["log"]
        assert_hash(interrupted_log, interrupted["log_sha256"])
        assert_hash(replacement_log, replacement["log_sha256"])
        interrupted_epochs = validate_epoch_log(
            interrupted_log,
            terminal_epoch=interrupted["last_logged_epoch"],
            completed=False,
        )
        replacement_epochs = validate_epoch_log(
            replacement_log,
            terminal_epoch=EXPECTED_EPOCHS,
            completed=True,
        )
        interrupted_run_dir = root / interrupted["wandb_run_dir"]
        replacement_run_dir = root / replacement["wandb_run_dir"]
        interrupted_run_file = interrupted_run_dir / f"run-{interrupted['wandb_id']}.wandb"
        replacement_run_file = replacement_run_dir / f"run-{replacement['wandb_id']}.wandb"
        assert_hash(interrupted_run_file, interrupted["wandb_file_sha256"])
        assert_hash(replacement_run_file, replacement["wandb_file_sha256"])
        expected_scratch_files = {
            "files/output.log",
            "files/requirements.txt",
            "files/wandb-metadata.json",
            "logs/debug-internal.log",
            "logs/debug.log",
            f"run-{interrupted['wandb_id']}.wandb",
        }
        scratch_files = {
            path.relative_to(interrupted_run_dir).as_posix()
            for path in interrupted_run_dir.rglob("*")
            if path.is_file() and not path.is_symlink()
        }
        require(scratch_files == expected_scratch_files, f"interrupted W&B scratch manifest differs for {key}")
        require(not (interrupted_run_dir / "files/wandb-summary.json").exists(), f"interrupted W&B summary unexpectedly exists for {key}")
        require(attempt_index[key].get("completed_at") == replacement["ledger_completed_at"], f"known completion time differs for {key}")
        require(attempt_index[key].get("artifact_sha256") == replacement["artifact_sha256"], f"known replacement artifact map differs for {key}")
        expected_attempt_log = str((root / replacement["log"]).resolve())
        require(attempt_index[key].get("log") == expected_attempt_log, f"known replacement log receipt differs for {key}")
        directory = Path(run_index[key]["directory"]).resolve()
        receipt = load_json(directory / "wandb_run.json")
        require(receipt.get("run_id") == replacement["wandb_id"], f"known replacement W&B ID differs for {key}")
        require(receipt.get("state") == "finished", f"known replacement W&B receipt state differs for {key}")
        output.append({
            "cell": dict(cell),
            "command_sha256": expected["command_sha256"],
            "interrupted_attempt": {
                "log": file_receipt(interrupted_log, root),
                **interrupted_epochs,
                "core_artifacts_present_at_interruption": False,
                "interrupted_directory_contained_only_wandb_work_state": True,
                "wandb": {
                    "run_id": interrupted["wandb_id"],
                    "state": "crashed",
                    "created_at": interrupted["wandb_created_at"],
                    "heartbeat_at": interrupted["wandb_heartbeat_at"],
                    "history_lines": interrupted["wandb_history_lines"],
                    "scratch_run_file": file_receipt(interrupted_run_file, root),
                    "scratch_manifest": sorted(scratch_files),
                },
            },
            "replacement_attempt": {
                "restart_policy": "restart_from_epoch_one",
                "log": file_receipt(replacement_log, root),
                **replacement_epochs,
                "ledger_completed_at": replacement["ledger_completed_at"],
                "artifact_sha256": dict(replacement["artifact_sha256"]),
                "wandb": {
                    "run_id": replacement["wandb_id"],
                    "state": "finished",
                    "created_at": replacement["wandb_created_at"],
                    "heartbeat_at": replacement["wandb_heartbeat_at"],
                    "history_lines": replacement["wandb_history_lines"],
                    "run_file": file_receipt(replacement_run_file, root),
                },
            },
        })
    return output


def query_remote_wandb(lineages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    try:
        import wandb  # type: ignore
    except ImportError as exc:
        raise AuditError("wandb SDK is required for remote terminal-state verification") from exc
    observed_from = utc_now()
    api = wandb.Api(timeout=30)
    runs: list[dict[str, Any]] = []
    for lineage in lineages:
        for phase, expected in (
            ("interrupted", lineage["interrupted_attempt"]["wandb"]),
            ("replacement", lineage["replacement_attempt"]["wandb"]),
        ):
            run_id = expected["run_id"]
            try:
                run = api.run(f"crlc112358/lewm-memory-popgym/{run_id}")
            except Exception as exc:
                raise AuditError(f"cannot query W&B run {run_id}: {exc}") from exc
            attrs = getattr(run, "_attrs", {})
            state = str(run.state)
            created_at = attrs.get("createdAt") or getattr(run, "created_at", None)
            heartbeat_at = attrs.get("heartbeatAt")
            history_lines = attrs.get("historyLineCount")
            require(state == expected["state"], f"remote W&B state differs for {run_id}: {state}")
            require(created_at == expected["created_at"], f"remote W&B createdAt differs for {run_id}")
            require(heartbeat_at == expected["heartbeat_at"], f"remote W&B heartbeatAt differs for {run_id}")
            require(history_lines == expected["history_lines"], f"remote W&B history count differs for {run_id}")
            runs.append({
                "cell": dict(lineage["cell"]),
                "phase": phase,
                "run_id": run_id,
                "state": state,
                "created_at": created_at,
                "heartbeat_at": heartbeat_at,
                "history_lines": history_lines,
                "url": f"https://wandb.ai/crlc112358/lewm-memory-popgym/runs/{run_id}",
            })
    observed_through = utc_now()
    return {
        "entity": "crlc112358",
        "project": "lewm-memory-popgym",
        "observed_from": observed_from,
        "observed_through": observed_through,
        "runs": runs,
    }


def verify_resume_counts(attempts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    completed = [parse_utc(str(row.get("completed_at"))) for row in attempts]
    first_group = [
        lineage["replacement"]["ledger_completed_at"]
        for lineage in LINEAGES[:4]
    ]
    first_start, first_end = map(parse_utc, (min(first_group), max(first_group)))
    first_before = sum(value < first_start for value in completed)
    first_after = sum(value <= first_end for value in completed)
    second_time = parse_utc(LINEAGES[4]["replacement"]["ledger_completed_at"])
    second_before = sum(value < second_time for value in completed)
    second_after = sum(value <= second_time for value in completed)
    require((first_before, first_after) == (136, 140), "first resume count chronology differs")
    require((second_before, second_after) == (180, 181), "second resume count chronology differs")
    events = [
        {
            "resume_index": 1,
            "command": "scripts/run_lewm_v8_v18.py --resume",
            "pre_resume_counts": {
                "planned": EXPECTED_CELLS,
                "valid_complete": 136,
                "absent": 64,
                "partial_or_invalid_core": 0,
                "interrupted_wandb_scratch_cells": 4,
            },
            "replacement_cell_count": 4,
            "post_replacement_counts": {"valid_complete": 140, "absent": 60},
            "resume_policy": "complete_cell_only",
        },
        {
            "resume_index": 2,
            "command": "scripts/run_lewm_v8_v18.py --resume",
            "runner_started_at": "2026-07-02T15:04:57Z",
            "lock_observation": {
                "evidence_class": "operator_transcript",
                "created_at": "2026-07-02T15:05:16.222019+00:00",
                "pid": 3171896,
                "sha256": "68eee1e8e357b0696c24a54f5b3fb80dd2f8de834417a11d2af26f13687aabbf",
            },
            "pre_resume_counts": {
                "planned": EXPECTED_CELLS,
                "valid_complete": 180,
                "absent": 20,
                "partial_or_invalid_core": 0,
                "interrupted_wandb_scratch_cells": 1,
            },
            "replacement_cell_count": 1,
            "post_replacement_counts": {"valid_complete": 181, "absent": 19},
            "resume_policy": "complete_cell_only",
        },
    ]
    assert_resume_events(events)
    return events


def assert_resume_events(events: Any) -> None:
    require(isinstance(events, list), "resume events must be a list, not null")
    require(len(events) == 2, "restart audit must contain exactly two resume events")
    require([event.get("resume_index") for event in events] == [1, 2], "resume event indices differ")
    expected = (
        ({"planned": 200, "valid_complete": 136, "absent": 64,
          "partial_or_invalid_core": 0, "interrupted_wandb_scratch_cells": 4},
         {"valid_complete": 140, "absent": 60}),
        ({"planned": 200, "valid_complete": 180, "absent": 20,
          "partial_or_invalid_core": 0, "interrupted_wandb_scratch_cells": 1},
         {"valid_complete": 181, "absent": 19}),
    )
    for event, (pre, post) in zip(events, expected, strict=True):
        require(event.get("pre_resume_counts") == pre, "resume pre-count receipt differs")
        require(event.get("post_replacement_counts") == post, "resume post-count receipt differs")


def verify_parent_only_pause(
    attempt_index: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    event = json.loads(json.dumps(PARENT_ONLY_PAUSE))
    key = (
        str(event["cell"]["task"]),
        str(event["cell"]["design"]),
        int(event["cell"]["seed"]),
    )
    require(key == ("stacker.stack_4", "vicreg_hacssmv8_static", 18003), "parent pause cell differs")
    require(event["classification"] == "runner_parent_pause_only", "parent pause classification differs")
    require(event["trainer_signaled"] is False, "parent pause incorrectly signals trainer")
    require(event["trainer_continued"] is True, "parent pause trainer-continuation fact differs")
    require(event["trainer_interrupted"] is False, "parent pause incorrectly marks trainer interrupted")
    require(event["restart_required"] is False, "parent pause incorrectly requires restart")
    require(event["cell_count_effect"] == 0, "parent pause count effect differs")
    ordered = [
        event["sigstop_at_approximately"],
        event["stopped_state_observed_at"],
        LINEAGES[4]["replacement"]["wandb_heartbeat_at"],
        event["trainer_exited_cleanly_at"],
        event["gpu_competitor_released_at"],
        event["sigcont_at_approximately"],
        event["parent_committed_completed_child_at"],
    ]
    parsed = [parse_utc(value) for value in ordered]
    require(parsed == sorted(parsed) and len(set(parsed)) == len(parsed), "parent pause chronology differs")
    require(
        attempt_index[key].get("completed_at")
        == event["parent_committed_completed_child_at"],
        "parent pause ledger-commit receipt differs",
    )
    return event


def bound_receipts(root: Path) -> dict[str, dict[str, Any]]:
    paths = {
        "protocol": root / PROTOCOL_REL,
        "summary": root / SUMMARY_REL,
        "runs": root / RUNS_REL,
        "attempts": root / ATTEMPTS_REL,
        "analysis": root / ANALYSIS_REL,
        "cells_csv": root / CELLS_CSV_REL,
        "contrasts_csv": root / CONTRASTS_CSV_REL,
    }
    return {name: file_receipt(path, root) for name, path in paths.items()}


def build_audit(root: Path) -> dict[str, Any]:
    assert_terminal_quiescence(root)
    protocol = load_json(root / PROTOCOL_REL)
    commands, protocol_receipt = validate_protocol(protocol, root / PROTOCOL_REL)
    summary = load_json(root / SUMMARY_REL)
    summary_receipt = validate_summary(summary)
    runs = load_json(root / RUNS_REL)
    attempts = load_json(root / ATTEMPTS_REL)
    run_index, attempt_index, artifact_check = validate_ledgers(
        root, commands, runs, attempts
    )
    manifest_hash = artifact_manifest_sha256(run_index.values())
    analysis = load_json(root / ANALYSIS_REL)
    analysis_receipt = validate_analysis(
        root, analysis, EXPECTED_PROTOCOL_SHA256, manifest_hash
    )
    supersedes = verify_stale_audit(root)
    lineages = verify_local_lineages(
        root, commands, run_index, attempt_index
    )
    resume_events = verify_resume_counts(attempts)
    parent_only_pause = verify_parent_only_pause(attempt_index)
    remote_wandb = query_remote_wandb(lineages)

    runner_log = file_receipt(root / ORIGINAL_RUNNER_LOG_REL, root)
    first_resume_log = file_receipt(root / FIRST_RESUME_LOG_REL, root)
    require(runner_log["size_bytes"] == 0 and runner_log["sha256"] == EMPTY_SHA256, "original runner log is no longer the known empty receipt")
    require(first_resume_log["size_bytes"] == 0 and first_resume_log["sha256"] == EMPTY_SHA256, "first resume wrapper log is no longer the known empty receipt")

    receipts_before = bound_receipts(root)
    assert_terminal_quiescence(root)
    receipts_after = bound_receipts(root)
    require(
        {key: value["sha256"] for key, value in receipts_before.items()}
        == {key: value["sha256"] for key, value in receipts_after.items()},
        "bound V18 receipts changed during audit generation",
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "study": "lewm-v8-v18-confirmation",
        "scope": {
            "included": [
                "runner_interruptions",
                "complete_cell_restarts",
                "wandb_terminal_state",
                "artifact_provenance",
                "final_runner_and_analysis_receipts",
            ],
            "excluded": ["scientific_metric_values", "scientific_metric_interpretation"],
        },
        "generated_at": utc_now(),
        "time_basis": {
            "canonical": "UTC",
            "display_timezone": "Asia/Taipei",
            "operator_transcript_times_are_approximate_where_marked": True,
        },
        "generator": {
            "path": str(Path(__file__).resolve()),
            "sha256": file_sha256(Path(__file__).resolve()),
        },
        "supersedes": supersedes,
        "protocol": protocol_receipt,
        "terminal_preconditions": {
            "runner_trainer_analyzer_processes_active": 0,
            "runner_lock_absent": True,
            "summary": summary_receipt,
            "analysis": analysis_receipt,
        },
        "bound_receipts": receipts_after,
        "artifact_binding": {
            **artifact_check,
            "artifact_manifest_sha256": manifest_hash,
            "artifact_manifest_bound_by_analysis": True,
            "artifact_bytes_hashed_without_parsing_scientific_content": True,
        },
        "runner_receipts": {
            "original_runner_log": runner_log,
            "first_resume_wrapper_log": first_resume_log,
            "caveat": (
                "Both runner-facing text receipts are zero-byte files; terminal "
                "claims are therefore bound to ledgers, cell logs, artifacts, and W&B."
            ),
        },
        "resume_events": resume_events,
        "attempt_lineages": lineages,
        "remote_wandb_terminal_observation": remote_wandb,
        "non_restart_events": [parent_only_pause],
        "final_study_snapshot": {
            "status": "COMPLETE",
            "planned_cells": EXPECTED_CELLS,
            "completed_valid_cells": EXPECTED_CELLS,
            "absent_cells": 0,
            "failed_or_invalid_cells": 0,
            "terminal_attempt_ledger_rows": EXPECTED_CELLS,
            "finished_local_wandb_receipts": EXPECTED_CELLS,
            "analysis_complete": True,
        },
        "assertions": [
            "All five interrupted cells lacked core artifacts and were restarted from epoch one.",
            "The five original W&B runs are crashed and the five replacement runs are finished.",
            "The parent-only SIGSTOP/SIGCONT event did not signal or interrupt the active trainer.",
            "The final frozen grid is COMPLETE 200/200 and bound to the write-once analysis bundle.",
        ],
    }


def atomic_write_json(path: Path, value: Mapping[str, Any], *, replace: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(f"output exists; pass --replace to regenerate: {path}")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists() and not replace:
            raise FileExistsError(f"output appeared during generation: {path}")
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def self_test() -> None:
    require(len(EXPECTED_CELL_SET) == EXPECTED_CELLS, "frozen cell-set cardinality differs")
    require(len(LINEAGES) == 5, "known lineage count differs")
    require(len({tuple(value["cell"].values()) for value in LINEAGES}) == 5, "known lineages duplicate")
    for lineage in LINEAGES:
        require(HEX64_RE.fullmatch(lineage["command_sha256"]) is not None, "bad known command hash")
        for phase in ("interrupted", "replacement"):
            require(HEX64_RE.fullmatch(lineage[phase]["log_sha256"]) is not None, "bad known log hash")
            require(HEX64_RE.fullmatch(lineage[phase]["wandb_file_sha256"]) is not None, "bad known W&B hash")
        require(set(lineage["replacement"]["artifact_sha256"]) == set(CORE_ARTIFACTS), "known artifact map differs")
        require(all(HEX64_RE.fullmatch(value) for value in lineage["replacement"]["artifact_sha256"].values()), "bad known artifact hash")
    require(PARENT_ONLY_PAUSE["classification"] == "runner_parent_pause_only", "parent-only pause classification differs")
    require(PARENT_ONLY_PAUSE["trainer_signaled"] is False, "parent-only pause trainer signal differs")
    require(PARENT_ONLY_PAUSE["trainer_interrupted"] is False, "parent-only pause interruption differs")
    require(PARENT_ONLY_PAUSE["restart_required"] is False, "parent-only pause restart differs")
    synthetic_attempts = (
        [{"completed_at": "2026-07-02T00:00:00Z"}] * 136
        + [{"completed_at": lineage["replacement"]["ledger_completed_at"]}
           for lineage in LINEAGES[:4]]
        + [{"completed_at": "2026-07-02T02:00:00Z"}] * 40
        + [{"completed_at": LINEAGES[4]["replacement"]["ledger_completed_at"]}]
        + [{"completed_at": "2026-07-02T16:00:00Z"}] * 19
    )
    resume_events = verify_resume_counts(synthetic_attempts)
    assert_resume_events(resume_events)
    require(
        [event["resume_index"] for event in resume_events] == [1, 2],
        "resume-count helper did not return both events",
    )
    require(json_sha256({"b": 2, "a": 1}) == hashlib.sha256(b'{"a":1,"b":2}').hexdigest(), "canonical JSON hash differs")
    with tempfile.TemporaryDirectory(prefix="v18-restart-audit-selftest-") as directory:
        path = Path(directory) / "audit.json"
        atomic_write_json(path, {"schema_version": 2}, replace=False)
        require(load_json(path) == {"schema_version": 2}, "atomic write/read differs")
        try:
            atomic_write_json(path, {"schema_version": 3}, replace=False)
        except FileExistsError:
            pass
        else:
            raise AuditError("atomic no-replace guard failed")
        atomic_write_json(path, {"schema_version": 2, "replaced": True}, replace=True)
        require(load_json(path).get("replaced") is True, "atomic replace differs")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        print("V18 restart-audit helper self-test: PASS")
        return
    if args.output is None:
        raise SystemExit("--output is required unless --self-test is used")
    root = args.repo_root.resolve()
    require((root / "scripts/run_lewm_v8_v18.py").is_file(), f"not the V18 repository root: {root}")
    output = args.output if args.output.is_absolute() else root / args.output
    output = output.resolve()
    protected = {
        (root / rel).resolve()
        for rel in (
            PROTOCOL_REL,
            RUNS_REL,
            ATTEMPTS_REL,
            SUMMARY_REL,
            ANALYSIS_REL,
            CELLS_CSV_REL,
            CONTRASTS_CSV_REL,
            STALE_AUDIT_REL,
        )
    }
    require(output not in protected, f"output would overwrite an input receipt: {output}")
    audit = build_audit(root)
    assert_terminal_quiescence(root)
    atomic_write_json(output, audit, replace=args.replace)
    print(json.dumps({
        "output": str(output),
        "sha256": file_sha256(output),
        "schema_version": SCHEMA_VERSION,
        "completed_valid_cells": EXPECTED_CELLS,
    }, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except AuditError as exc:
        raise SystemExit(f"V18 restart audit refused: {exc}") from exc
