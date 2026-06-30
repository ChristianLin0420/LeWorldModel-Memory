#!/usr/bin/env python3
"""Independent read-only integrity and gate audit for the CF-HIRO-v13 screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
TASKS = (
    "cartpole.swingup", "fish.swim", "pendulum.swingup", "walker.walk")
DESIGNS = (
    "cfhirov13", "cfhirov13_fullanchor", "cfhirov13_triangular",
    "cfhirov13_noshrink", "cfhirov13_noaction", "cfhirov13_nocorrect",
    "ssm", "hacssmv8", "kdiov11")
CF_DESIGNS = DESIGNS[:6]
CONTROLS = CF_DESIGNS[1:]
SEED = 13_001
EPOCHS = 30
EXPECTED_CELLS = 36
STUDY = "hacssm-v13-screen-cfhiro30"
ENTITY = "crlc112358"
PROJECT = "lewm-memory-popgym"
V11_RANKING = "rawdiff_displacement_detached"
PRIMARY = "heldout_prior_state_nmse"
SOURCE_MANIFEST = (
    "lewm/models/cf_hiro.py",
    "lewm/models/memory_model.py",
    "lewm/models/memory.py",
    "lewm/models/leworldmodel.py",
    "lewm/models/encoder.py",
    "scripts/train_cf_hiro_v13.py",
    "scripts/run_cf_hiro_v13_screen.py",
    "scripts/analyze_cf_hiro_v13_screen.py",
    "scripts/audit_cf_hiro_v13_screen.py",
    "scripts/train_siro_v12.py",
    "scripts/train_hacssm_v11.py",
    "scripts/train_hacssm_v10.py",
    "scripts/hacssm_v11_data.py",
)


class AuditFailure(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AuditFailure(f"missing JSON file: {path}")
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditFailure(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditFailure(f"JSON root must be an object: {path}")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise AuditFailure(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AuditFailure(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise AuditFailure(f"{label} is not finite: {value!r}")
    return result


def _require(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise AuditFailure(f"{label}={actual!r}; expected {expected!r}")


def run_name(task: str, design: str) -> str:
    suffix = f"-rank-{V11_RANKING}" if design == "kdiov11" else ""
    return f"lewm-dmc:{task}-{design}-s{SEED}{suffix}"


def _resolve(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise AuditFailure(f"invalid manifest path {value!r}")
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT):
        raise AuditFailure(f"manifest path escapes repository: {value}")
    return resolved


def _flag(command: Sequence[Any], flag: str) -> str:
    indices = [index for index, value in enumerate(command) if value == flag]
    if len(indices) != 1 or indices[0] + 1 >= len(command):
        raise AuditFailure(f"command must contain exactly one {flag}")
    return str(command[indices[0] + 1])


def validate_protocol(root: Path) -> dict[str, Any]:
    protocol = _load_json(root / "screen_protocol.json")
    exact = {
        "schema_version": 1,
        "scope": "excluded_adaptive_v13_screen_after_failed_v12",
        "seed": SEED,
        "tasks": list(TASKS),
        "designs": list(DESIGNS),
        "runs": EXPECTED_CELLS,
        "epochs": EPOCHS,
        "study": STUDY,
        "wandb_entity": ENTITY,
        "wandb_project": PROJECT,
        "v11_comparator_action_ranking": V11_RANKING,
        "blas_threads_per_process": 4,
        "automatic_100_epoch_launch_in_this_process": False,
    }
    for key, expected in exact.items():
        _require(protocol.get(key), expected, f"protocol.{key}")
    _require(protocol.get("git_branch"), "learnable-memory", "protocol.git_branch")
    commit = protocol.get("git_commit")
    if (not isinstance(commit, str) or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)):
        raise AuditFailure("protocol.git_commit is not a full lowercase SHA-1")
    _require(protocol.get("git_upstream_commit"), commit, "protocol.git_upstream_commit")
    _require(protocol.get("git_worktree_clean"), True, "protocol.git_worktree_clean")
    _require(protocol.get("git_head_pushed"), True, "protocol.git_head_pushed")
    _require(protocol.get("gpus"), ["0", "1", "2", "3"], "protocol.gpus")
    _require(
        protocol.get("task_pinned_gpu"), dict(zip(TASKS, ("0", "1", "2", "3"), strict=True)),
        "protocol.task_pinned_gpu")
    _require(
        set(protocol.get("source_sha256", {})), set(SOURCE_MANIFEST),
        "protocol source manifest")
    for relative, expected in protocol["source_sha256"].items():
        path = _resolve(relative)
        if not path.is_file():
            raise AuditFailure(f"missing source {relative}")
        _require(sha256_file(path), expected, f"source hash {relative}")
    data = protocol.get("data")
    if not isinstance(data, Mapping) or set(data) != set(TASKS):
        raise AuditFailure("protocol data manifest differs from frozen tasks")
    for task in TASKS:
        for split in ("train", "val"):
            path = _resolve(data[task].get(split))
            if not path.is_file():
                raise AuditFailure(f"missing data {task}/{split}")
            _require(
                sha256_file(path), data[task].get(f"{split}_sha256"),
                f"data hash {task}/{split}")
    commands = protocol.get("commands")
    if not isinstance(commands, Mapping) or set(commands) != set(TASKS):
        raise AuditFailure("protocol command grid differs")
    for task in TASKS:
        if not isinstance(commands[task], list) or len(commands[task]) != len(DESIGNS):
            raise AuditFailure(f"protocol commands for {task} are incomplete")
        for design, command in zip(DESIGNS, commands[task], strict=True):
            if not isinstance(command, list):
                raise AuditFailure(f"command {task}/{design} is not a list")
            expected_flags = {
                "--memory-mode": design, "--seed": str(SEED),
                "--epochs": str(EPOCHS), "--batch-size": "64",
                "--lr": "0.0003", "--weight-decay": "0.00001",
                "--embed-dim": "128", "--history-len": "3",
                "--wandb-entity": ENTITY, "--wandb-project": PROJECT,
                "--wandb-mode": "online", "--wandb-study": STUDY,
            }
            for flag, expected in expected_flags.items():
                _require(_flag(command, flag), expected, f"command {task}/{design} {flag}")
            if "--wandb" not in command:
                raise AuditFailure(f"command {task}/{design} omits --wandb")
    return protocol


def _load_checkpoint(path: Path, label: str) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise AuditFailure(f"{label}: cannot load model.pt: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditFailure(f"{label}: model.pt is not a dictionary")
    return value


def validate_cell(
        root: Path, task: str, design: str,
        protocol: Mapping[str, Any]) -> dict[str, Any]:
    directory = root / run_name(task, design)
    label = f"{task}/{design}"
    paths = {name: directory / name for name in (
        "model.pt", "metrics.json", "eval_rollout.npz", "wandb_run.json")}
    for path in paths.values():
        if not path.is_file():
            raise AuditFailure(f"{label}: missing {path.name}")
    metrics = _load_json(paths["metrics.json"])
    for key, expected in {
            "env": f"dmc:{task}", "design": design,
            "seed": SEED, "epochs": EPOCHS}.items():
        _require(metrics.get(key), expected, f"{label}.metrics.{key}")
    for key in (
            PRIMARY, "clean_prior_state_nmse", "initial_encoder_integrator_probe_nmse",
            "predictive_loss_convergence_relative_change", "encoder_mean_channel_variance",
            "encoder_covariance_effective_rank", "encoder_singleton_max_abs",
            "encoder_prefix_max_abs"):
        _finite(metrics.get(key), f"{label}.{key}")
    for split in ("train", "val"):
        _require(
            metrics.get(f"{split}_data_sha256"),
            protocol["data"][task][f"{split}_sha256"], f"{label}.{split} hash")
    rollout_hash = sha256_file(paths["eval_rollout.npz"])
    _require(metrics.get("eval_rollout_sha256"), rollout_hash, f"{label}.rollout hash")
    wandb = _load_json(paths["wandb_run.json"])
    for key, expected in {
            "state": "finished", "mode": "online", "study": STUDY,
            "entity": ENTITY, "project": PROJECT,
            "eval_rollout_sha256": rollout_hash}.items():
        _require(wandb.get(key), expected, f"{label}.wandb.{key}")
    if not wandb.get("run_id") or not wandb.get("url"):
        raise AuditFailure(f"{label}: incomplete W&B identity")
    _require(wandb.get("run_name"), f"{STUDY}-{directory.name}", f"{label}.run name")
    checkpoint = _load_checkpoint(paths["model.pt"], label)
    args = checkpoint.get("args")
    if not isinstance(args, Mapping):
        raise AuditFailure(f"{label}: checkpoint args are missing")
    for key, expected in {
            "memory_mode": design, "seed": SEED, "epochs": EPOCHS,
            "wandb": True, "wandb_entity": ENTITY, "wandb_project": PROJECT,
            "wandb_mode": "online", "wandb_study": STUDY,
            "eval_rollout_episode": 0}.items():
        _require(args.get(key), expected, f"{label}.args.{key}")
    if checkpoint.get("final_metrics") != metrics:
        raise AuditFailure(f"{label}: checkpoint final metrics differ from metrics.json")
    history = checkpoint.get("history")
    if not isinstance(history, list) or len(history) != EPOCHS:
        raise AuditFailure(f"{label}: checkpoint history is not 30 rows")
    epochs = [row.get("epoch") for row in history if isinstance(row, Mapping)]
    if epochs != list(range(1, EPOCHS + 1)) or len(set(epochs)) != EPOCHS:
        raise AuditFailure(f"{label}: W&B epoch identities are not 30 unique rows")
    if design in CF_DESIGNS:
        fits = checkpoint.get("fit_history")
        final = checkpoint.get("final_operator_fit")
        if not isinstance(fits, list) or len(fits) != EPOCHS + 1:
            raise AuditFailure(f"{label}: fit history is not 31 rows")
        if not isinstance(final, Mapping) or not isinstance(final.get("receipts"), Mapping):
            raise AuditFailure(f"{label}: final fit is missing")
        _require(final["receipts"].get("fit_index"), EPOCHS, f"{label}.fit index")
        state = checkpoint.get("model_state_dict")
        if not isinstance(state, Mapping):
            raise AuditFailure(f"{label}: model state is missing")
        prefix = "world.mem_cfhirov13."
        for name in (
                "state_matrix", "action_matrix", "read_matrix", "process_covariance",
                "measurement_covariance", "initial_covariance", "steady_prior_covariance",
                "steady_gain", "initial_map", "output_mean", "action_mean"):
            if (not isinstance(state.get(prefix + name), torch.Tensor)
                    or not isinstance(final.get(name), torch.Tensor)
                    or not torch.equal(state[prefix + name].cpu(), final[name].cpu())):
                raise AuditFailure(f"{label}: serialized {name} differs from final fit")
    if design == "kdiov11":
        _require(
            metrics.get("development_action_ranking"), V11_RANKING,
            f"{label}.KDIO ranking")
    return {
        "task": task, "design": design, "metrics": metrics,
        "wandb_run_id": wandb["run_id"], "wandb_url": wandb["url"],
        "artifact_sha256": {name: sha256_file(path) for name, path in paths.items()},
    }


def validate_runner(root: Path, rows: Sequence[Mapping[str, Any]], protocol) -> None:
    if (root / ".cf_hiro_v13_screen.lock").exists():
        raise AuditFailure("runner lock still exists")
    path = root / "screen_runs.json"
    if not path.is_file():
        raise AuditFailure("missing screen_runs.json")
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != EXPECTED_CELLS:
        raise AuditFailure("screen_runs.json is not 36 records")
    row_map = {(row["task"], row["design"]): row for row in rows}
    seen = set()
    for record in records:
        pair = (record.get("task"), record.get("design"))
        if pair in seen or pair not in row_map:
            raise AuditFailure(f"invalid runner record pair {pair}")
        seen.add(pair)
        task, design = pair
        _require(str(record.get("gpu")), protocol["task_pinned_gpu"][task],
                 f"runner GPU {task}/{design}")
        if _finite(record.get("seconds"), f"runner seconds {task}/{design}") <= 0:
            raise AuditFailure(f"runner seconds are non-positive: {task}/{design}")
        _require(
            record.get("artifact_sha256"), row_map[pair]["artifact_sha256"],
            f"runner artifact hashes {task}/{design}")
    _require(seen, set(row_map), "runner cell set")


def _values(rows, design, key):
    mapping = {row["task"]: float(row["metrics"][key])
               for row in rows if row["design"] == design}
    if set(mapping) != set(TASKS):
        raise AuditFailure(f"incomplete {design}/{key}")
    return np.asarray([mapping[task] for task in TASKS])


def _contrast(rows, reference):
    candidate, baseline = _values(rows, "cfhirov13", PRIMARY), _values(
        rows, reference, PRIMARY)
    return ((baseline.mean() - candidate.mean()) / baseline.mean(),
            int((candidate < baseline).sum()))


def recompute_gates(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    representation_failures = []
    for row in rows:
        if row["design"] != "cfhirov13":
            continue
        m, label = row["metrics"], f"{row['task']}/{row['design']}"
        if m["encoder_mean_channel_variance"] < 1e-5:
            representation_failures.append(f"{label}: variance")
        if m["encoder_covariance_effective_rank"] < 16:
            representation_failures.append(f"{label}: rank")
        if abs(m["encoder_singleton_max_abs"]) > 1e-5:
            representation_failures.append(f"{label}: singleton")
        if abs(m["encoder_prefix_max_abs"]) > 1e-5:
            representation_failures.append(f"{label}: prefix")
    numerical_failures = []
    boundary = 1.0 - math.sqrt(np.finfo(np.float32).eps)
    for row in rows:
        if row["design"] not in CF_DESIGNS:
            continue
        m, design = row["metrics"], row["design"]
        label = f"{row['task']}/{design}"
        if m.get("fit_updates") != 31 or m.get("cf_hiro_fit_fit_index") != 30:
            numerical_failures.append(f"{label}: fit schedule")
        for key, maximum in {
                "cf_hiro_streaming_max_abs": 1e-5,
                "cf_hiro_projector_algebra_max_abs": 1e-5,
                "cf_hiro_initial_reconstruction_max_abs": 1e-5,
                "cf_hiro_complement_dynamic_orthogonality_max_abs": 1e-5,
                "cf_hiro_core_steady_riccati_relative_residual": 1e-6}.items():
            if abs(_finite(m.get(key), f"{label}.{key}")) > maximum:
                numerical_failures.append(f"{label}: {key}")
        if _finite(m.get("cf_hiro_core_state_spectral_radius"), label) > boundary + 2e-6:
            numerical_failures.append(f"{label}: radius")
        if design != "cfhirov13_triangular" and m.get(
                "cf_hiro_core_state_is_real_normal_contraction") is not True:
            numerical_failures.append(f"{label}: normality")
    full = {row["task"]: row["metrics"] for row in rows if row["design"] == "cfhirov13"}
    external = {}
    for reference in ("ssm", "hacssmv8", "kdiov11"):
        reduction, wins = _contrast(rows, reference)
        external[reference] = reduction >= .05 and wins >= 3
    cand = np.asarray([full[t][PRIMARY] for t in TASKS])
    integ = np.asarray([full[t]["initial_encoder_integrator_probe_nmse"] for t in TASKS])
    external["integrator"] = (
        (integ.mean() - cand.mean()) / integ.mean() >= .05
        and int((cand < integ).sum()) >= 3)
    internal = {}
    for control in CONTROLS:
        reduction, wins = _contrast(rows, control)
        internal[control] = reduction >= (
            .05 if control == "cfhirov13_noaction" else .02) and wins >= 3
    action_tasks = 0
    for task in TASKS:
        m = full[task]
        action_tasks += int(
            m["cf_hiro_fit_held_fold_action_r2_even_to_odd"] > 0
            and m["cf_hiro_fit_held_fold_action_r2_odd_to_even"] > 0
            and m["cf_hiro_true_action_suffix_advantage"] > 0
            and m["cf_hiro_action_pair_accuracy"] > .5)
    energy = all(
        full[t]["cf_hiro_complement_anchor_rms"] > 1e-8
        and full[t]["cf_hiro_dynamic_initial_rms"] > 1e-8 for t in TASKS)
    full_late = [
        full[t]["predictive_loss_convergence_relative_change"] for t in TASKS]
    all_late = [
        abs(row["metrics"]["predictive_loss_convergence_relative_change"])
        for row in rows]
    convergence = (
        all(value >= 0 for value in full_late)
        and max(map(abs, full_late)) < .05 and float(np.median(all_late)) < .03)
    scientific = bool(
        not representation_failures and not numerical_failures
        and all(external.values()) and all(internal.values())
        and action_tasks >= 3 and energy and convergence)
    return {
        "representation_passed": not representation_failures,
        "numerical_passed": not numerical_failures,
        "external_passed": all(external.values()),
        "internal_passed": all(internal.values()),
        "action_passed": action_tasks >= 3,
        "energy_passed": energy,
        "convergence_passed": convergence,
        "scientific_gate_passed": scientific,
        "representation_failures": representation_failures,
        "numerical_failures": numerical_failures,
    }


def audit_status(
        *, artifact_integrity: bool, analyzer_consistent: bool,
        scientific_gate: bool) -> tuple[str, bool]:
    if not artifact_integrity or not analyzer_consistent:
        return "FAIL_CLOSED", False
    return ("PASS_COMPLETE" if scientific_gate else "PASS_COMPLETE_NEGATIVE"), True


def audit(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    protocol = None
    try:
        protocol = validate_protocol(root)
    except (AuditFailure, OSError, ValueError) as exc:
        errors.append(str(exc))
    if protocol is not None:
        for task in TASKS:
            for design in DESIGNS:
                try:
                    rows.append(validate_cell(root, task, design, protocol))
                except (AuditFailure, OSError, ValueError) as exc:
                    errors.append(str(exc))
    artifact_integrity = protocol is not None and len(rows) == EXPECTED_CELLS and not errors
    if artifact_integrity:
        try:
            validate_runner(root, rows, protocol)
        except (AuditFailure, OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(str(exc))
            artifact_integrity = False
    gates = recompute_gates(rows) if len(rows) == EXPECTED_CELLS else None
    analyzer_consistent = False
    analyzer_status = None
    if artifact_integrity and gates is not None:
        try:
            analysis = _load_json(root / "screen_analysis.json")
            decision = _load_json(root / "screen_decision.json")
            expected_status = (
                "SCREEN_PASS_100E_MANIFEST" if gates["scientific_gate_passed"]
                else "SCREEN_NO_GO")
            _require(analysis.get("artifact_integrity_passed"), True,
                     "analysis artifact integrity")
            _require(analysis.get("representation_gate_passed"),
                     gates["representation_passed"], "analysis representation")
            _require(analysis.get("numerical_gate", {}).get("passed"),
                     gates["numerical_passed"], "analysis numerical")
            _require(analysis.get("external_performance_gate_passed"),
                     gates["external_passed"], "analysis external")
            _require(analysis.get("internal_mechanism_gate_passed"),
                     gates["internal_passed"], "analysis internal")
            _require(analysis.get("action_gate", {}).get("passed"),
                     gates["action_passed"], "analysis action")
            _require(analysis.get("direct_sum_energy_gate", {}).get("passed"),
                     gates["energy_passed"], "analysis energy")
            _require(analysis.get("convergence_gate", {}).get("passed"),
                     gates["convergence_passed"], "analysis convergence")
            _require(analysis.get("scientific_gate_passed"),
                     gates["scientific_gate_passed"], "analysis scientific")
            _require(analysis.get("status"), expected_status, "analysis status")
            _require(decision.get("status"), expected_status, "decision status")
            manifest = root / "contingent_100e_launch_manifest.json"
            _require(manifest.exists(), gates["scientific_gate_passed"],
                     "conditional manifest presence")
            if manifest.exists():
                continuation = _load_json(manifest)
                _require(continuation.get("runs"), 72, "continuation runs")
                _require(continuation.get("seeds"), [13002, 13003, 13004],
                         "continuation seeds")
                _require(continuation.get("automatic_launch_performed"), False,
                         "continuation auto-launch")
            analyzer_status = expected_status
            analyzer_consistent = True
        except (AuditFailure, OSError, ValueError) as exc:
            errors.append(str(exc))
    scientific = bool(gates and gates["scientific_gate_passed"])
    status, passed = audit_status(
        artifact_integrity=artifact_integrity,
        analyzer_consistent=analyzer_consistent,
        scientific_gate=scientific)
    return {
        "schema_version": 1,
        "scope": "independent_read_only_cf_hiro_v13_screen_audit",
        "root": str(root),
        "status": status,
        "passed": passed,
        "artifact_integrity_passed": artifact_integrity,
        "analyzer_receipt_consistent": analyzer_consistent,
        "scientific_gate_passed": scientific if gates is not None else None,
        "analyzer_status": analyzer_status,
        "expected_cells": EXPECTED_CELLS,
        "validated_cells": len(rows),
        "protocol_validated": protocol is not None,
        "recomputed_gates": gates,
        "errors": errors,
        "cells": [{
            "task": row["task"], "design": row["design"],
            "heldout_prior_state_nmse": row["metrics"][PRIMARY],
            "wandb_run_id": row["wandb_run_id"],
            "wandb_url": row["wandb_url"],
            "artifact_sha256": row["artifact_sha256"],
        } for row in rows],
    }


def audit_exit_code(report: Mapping[str, Any]) -> int:
    return 0 if report.get("passed") is True else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/hacssm_v13_screen_cfhiro30"))
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = args.root if args.root.is_absolute() else (ROOT / args.root).resolve()
    report = audit(root)
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    print(rendered, end="")
    if args.output is not None:
        output = args.output if args.output.is_absolute() else (ROOT / args.output).resolve()
        with output.open("x", encoding="utf-8") as stream:
            stream.write(rendered)
    raise SystemExit(audit_exit_code(report))


if __name__ == "__main__":
    main()
