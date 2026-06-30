#!/usr/bin/env python3
"""Run the frozen 36-cell CF-HIRO-v13 screen on four task-pinned GPUs."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.hacssm_v11_data import (
    DEFAULT_IMG_SIZE,
    DEFAULT_LENGTH,
    DEFAULT_TRAIN_EPISODES,
    DEFAULT_TRAIN_SEED,
    DEFAULT_VAL_EPISODES,
    DEFAULT_VAL_SEED,
    cache_name,
    sha256_file,
)
from scripts.train_cf_hiro_v13 import DESIGNS, V11_COMPARATOR_RANKING


TASKS = (
    "cartpole.swingup",
    "fish.swim",
    "pendulum.swingup",
    "walker.walk",
)
SEED = 13_001
DEFAULT_STUDY = "hacssm-v13-screen-cfhiro30"
DEFAULT_OUTPUT_ROOT = Path("outputs/hacssm_v13_screen_cfhiro30")
DEFAULT_LOG_ROOT = Path("logs/hacssm_v13_screen_cfhiro30")
DATA_ROOT = Path("outputs/hacssm_v11_data")
WANDB_ENTITY = "crlc112358"
WANDB_PROJECT = "lewm-memory-popgym"
BLAS_THREADS = 4
LOCK_NAME = ".cf_hiro_v13_screen.lock"

SOURCE_PATHS = (
    Path("lewm/models/cf_hiro.py"),
    Path("lewm/models/memory_model.py"),
    Path("lewm/models/memory.py"),
    Path("lewm/models/leworldmodel.py"),
    Path("lewm/models/encoder.py"),
    Path("scripts/train_cf_hiro_v13.py"),
    Path("scripts/run_cf_hiro_v13_screen.py"),
    Path("scripts/analyze_cf_hiro_v13_screen.py"),
    Path("scripts/audit_cf_hiro_v13_screen.py"),
    Path("scripts/train_siro_v12.py"),
    Path("scripts/train_hacssm_v11.py"),
    Path("scripts/train_hacssm_v10.py"),
    Path("scripts/hacssm_v11_data.py"),
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(task: str) -> str:
    return "dmc_" + task.replace(".", "_")


def _git_receipt() -> dict[str, object]:
    def value(*arguments: str) -> str:
        result = subprocess.run(
            ("git", *arguments), cwd=ROOT, check=True,
            text=True, capture_output=True)
        return result.stdout.strip()

    status = value("status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError(
            "refusing V13 evidence launch from a dirty worktree; commit and push first")
    head = value("rev-parse", "HEAD")
    upstream = value("rev-parse", "@{upstream}")
    if head != upstream:
        raise RuntimeError(
            f"refusing V13 evidence launch before push: HEAD {head} != upstream {upstream}")
    return {
        "git_branch": value("branch", "--show-current"),
        "git_commit": head,
        "git_upstream_commit": upstream,
        "git_worktree_clean": True,
        "git_head_pushed": True,
    }


def data_paths(task: str) -> tuple[Path, Path]:
    return (
        DATA_ROOT / cache_name(
            task, "train", DEFAULT_TRAIN_EPISODES, DEFAULT_LENGTH,
            DEFAULT_IMG_SIZE, DEFAULT_TRAIN_SEED),
        DATA_ROOT / cache_name(
            task, "val", DEFAULT_VAL_EPISODES, DEFAULT_LENGTH,
            DEFAULT_IMG_SIZE, DEFAULT_VAL_SEED),
    )


def run_name(task: str, design: str) -> str:
    suffix = (
        f"-rank-{V11_COMPARATOR_RANKING}" if design == "kdiov11" else "")
    return f"lewm-dmc:{task}-{design}-s{SEED}{suffix}"


def run_directory(output_root: Path, task: str, design: str) -> Path:
    return output_root / run_name(task, design)


def train_command(
        python: str, output_root: Path, study: str, epochs: int,
        task: str, design: str) -> list[str]:
    train_data, val_data = data_paths(task)
    return [
        python,
        str(ROOT / "scripts" / "train_cf_hiro_v13.py"),
        "--train-data", str(train_data),
        "--val-data", str(val_data),
        "--memory-mode", design,
        "--seed", str(SEED),
        "--epochs", str(epochs),
        "--output-dir", str(output_root),
        "--batch-size", "64",
        "--lr", "0.0003",
        "--weight-decay", "0.00001",
        "--num-workers", "2",
        "--img-size", "64",
        "--patch-size", "8",
        "--embed-dim", "128",
        "--encoder-layers", "6",
        "--encoder-heads", "4",
        "--predictor-layers", "4",
        "--predictor-heads", "8",
        "--history-len", "3",
        "--dropout", "0.1",
        "--sigreg-lambda", "0.1",
        "--sigreg-projections", "512",
        "--probe-ridge", "0.001",
        "--eval-target-key", "task_observation",
        "--corruption-seed", "11012",
        "--eval-rollout-episode", "0",
        "--device", "cuda",
        "--wandb",
        "--wandb-entity", WANDB_ENTITY,
        "--wandb-project", WANDB_PROJECT,
        "--wandb-mode", "online",
        "--wandb-study", study,
        "--extra-tag", "excluded-adaptive-screen,cf-hiro-v13",
    ]


def _validate_inputs() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for task in TASKS:
        train_path, val_path = data_paths(task)
        for path in (train_path, val_path):
            if not path.is_file():
                raise FileNotFoundError(f"missing frozen V13 screen cache {path}")
        result[task] = {
            "train": str(train_path),
            "train_sha256": sha256_file(train_path),
            "val": str(val_path),
            "val_sha256": sha256_file(val_path),
        }
    return result


def _run_task_queue(
        gpu: str, task: str, *, python: str, output_root: Path,
        log_root: Path, study: str, epochs: int) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for design in DESIGNS:
        command = train_command(python, output_root, study, epochs, task, design)
        log_path = log_root / f"{_slug(task)}-{design}-s{SEED}.log"
        if log_path.exists():
            raise FileExistsError(f"refusing to overwrite {log_path}")
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        environment["MUJOCO_GL"] = "egl"
        for variable in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
            environment[variable] = str(BLAS_THREADS)
        started = time.time()
        print(f"[gpu {gpu}] starting {task}/{design}", flush=True)
        with log_path.open("x", encoding="utf-8") as log:
            result = subprocess.run(
                command, cwd=ROOT, env=environment, stdout=log,
                stderr=subprocess.STDOUT, text=True, check=False)
        elapsed = time.time() - started
        if result.returncode:
            raise RuntimeError(
                f"{task}/{design} failed with status {result.returncode}; see {log_path}")
        directory = run_directory(output_root, task, design)
        required = tuple(
            directory / name for name in
            ("model.pt", "metrics.json", "eval_rollout.npz", "wandb_run.json"))
        for path in required:
            if not path.is_file():
                raise RuntimeError(f"{task}/{design} did not produce {path}")
        records.append({
            "gpu": gpu,
            "task": task,
            "design": design,
            "seconds": elapsed,
            "log": str(log_path),
            "metrics": str(directory / "metrics.json"),
            "artifact_sha256": {
                path.name: _file_sha256(path) for path in required},
        })
        print(f"[gpu {gpu}] finished {task}/{design} in {elapsed:.1f}s", flush=True)
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--study", default=DEFAULT_STUDY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--python", default=str(ROOT / ".venv" / "bin" / "python"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.output_root = (
        args.output_root if args.output_root.is_absolute()
        else (ROOT / args.output_root).resolve())
    args.log_root = (
        args.log_root if args.log_root.is_absolute()
        else (ROOT / args.log_root).resolve())
    gpu_ids = tuple(value.strip() for value in args.gpus.split(",") if value.strip())
    if gpu_ids != ("0", "1", "2", "3"):
        raise ValueError("the frozen CF-HIRO screen requires GPUs 0,1,2,3 in task order")
    if args.epochs != 30:
        raise ValueError("the frozen V13 screen requires exactly 30 epochs")
    if args.study != DEFAULT_STUDY:
        raise ValueError(f"the frozen V13 screen study is {DEFAULT_STUDY!r}")
    if not Path(args.python).is_file():
        raise FileNotFoundError(f"Python executable not found: {args.python}")
    data = _validate_inputs()
    commands = {
        task: [train_command(
            args.python, args.output_root, args.study, args.epochs, task, design)
            for design in DESIGNS]
        for task in TASKS}
    if args.dry_run:
        print(json.dumps({
            "gpus": gpu_ids,
            "tasks": TASKS,
            "designs": DESIGNS,
            "runs": len(TASKS) * len(DESIGNS),
            "epochs": args.epochs,
            "study": args.study,
            "output_root": str(args.output_root),
            "blas_threads_per_process": BLAS_THREADS,
            "commands": commands,
        }, indent=2))
        return

    git_receipt = _git_receipt()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output_root / "screen_protocol.json"
    lock_path = args.output_root / LOCK_NAME
    if protocol_path.exists() or lock_path.exists():
        raise FileExistsError(
            f"refusing to reuse CF-HIRO screen namespace {args.output_root}")
    missing_sources = [path for path in SOURCE_PATHS if not (ROOT / path).is_file()]
    if missing_sources:
        raise FileNotFoundError(f"missing V13 source-manifest files: {missing_sources}")
    protocol = {
        "schema_version": 1,
        "scope": "excluded_adaptive_v13_screen_after_failed_v12",
        "seed": SEED,
        "tasks": list(TASKS),
        "designs": list(DESIGNS),
        "runs": len(TASKS) * len(DESIGNS),
        "epochs": args.epochs,
        "gpus": list(gpu_ids),
        "task_pinned_gpu": dict(zip(TASKS, gpu_ids, strict=True)),
        "study": args.study,
        "wandb_entity": WANDB_ENTITY,
        "wandb_project": WANDB_PROJECT,
        "v11_comparator_action_ranking": V11_COMPARATOR_RANKING,
        "blas_threads_per_process": BLAS_THREADS,
        "blas_environment_variables": [
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS"],
        **git_receipt,
        "data": data,
        "source_sha256": {
            str(path): _file_sha256(ROOT / path) for path in SOURCE_PATHS},
        "commands": commands,
        "automatic_100_epoch_launch_in_this_process": False,
        "continuation_contract": (
            "analyzer writes a contingent 100e launch manifest only after every "
            "artifact, representation, mechanism, performance, action, and convergence "
            "gate passes; it never launches training"),
    }
    with protocol_path.open("x", encoding="utf-8") as stream:
        json.dump(protocol, stream, indent=2, sort_keys=True)
        stream.write("\n")
    lock_path.touch(exist_ok=False)
    try:
        all_records: list[dict[str, object]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(
                    _run_task_queue, gpu, task, python=args.python,
                    output_root=args.output_root, log_root=args.log_root,
                    study=args.study, epochs=args.epochs): task
                for task, gpu in zip(TASKS, gpu_ids, strict=True)}
            for future in concurrent.futures.as_completed(futures):
                all_records.extend(future.result())
        all_records.sort(key=lambda row: (
            TASKS.index(str(row["task"])), DESIGNS.index(str(row["design"]))))
        with (args.output_root / "screen_runs.json").open("x", encoding="utf-8") as stream:
            json.dump(all_records, stream, indent=2, sort_keys=True)
            stream.write("\n")
        if not args.skip_analysis:
            command = [
                args.python, str(ROOT / "scripts" / "analyze_cf_hiro_v13_screen.py"),
                "--root", str(args.output_root), "--epochs", str(args.epochs),
                "--study", args.study, "--seed", str(SEED), "--write"]
            result = subprocess.run(command, cwd=ROOT, check=False)
            if result.returncode:
                raise RuntimeError(
                    f"CF-HIRO screen analyzer reported artifact failure "
                    f"with status {result.returncode}")
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
