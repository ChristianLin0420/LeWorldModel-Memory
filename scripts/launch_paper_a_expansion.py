#!/usr/bin/env python3
"""Parallel launcher for the three preregistered Paper-A expansion waves."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
TASKS = ("t1", "t3")


@dataclass(frozen=True)
class Job:
    name: str
    command: tuple[str, ...]
    done_file: Path


def frozen_jobs(output: Path) -> list[Job]:
    jobs = []
    for task in TASKS:
        for arm in ("none", "gru", "lstm", "ssm", "fixed_trust"):
            for seed in range(5):
                directory = output / "frozen_swap" / task / arm / f"s{seed}"
                command = (
                    PYTHON, "scripts/train_frozen_official_swap.py",
                    "--task", task, "--arm", arm, "--seed", str(seed),
                    "--cache-root", str(output / "cache"),
                    "--weights", str(output / "pretrained/lewm-reacher/weights.pt"),
                    "--output", str(output / "frozen_swap"),
                    "--epochs", "100", "--batch-size", "64",
                    "--lr", "0.0003", "--weight-decay", "0.00001",
                    "--device", "cuda",
                )
                jobs.append(Job(f"{task}_{arm}_s{seed}", command,
                                directory / "metrics.json"))
    return jobs


def context_jobs(output: Path) -> list[Job]:
    jobs = []
    for task in TASKS:
        for history in (3, 16, 32, 56):
            for seed in range(3):
                directory = output / "long_context" / task / f"h{history}" / f"s{seed}"
                command = (
                    PYTHON, "scripts/train_official_long_context.py",
                    "--train-cache", str(output / f"cache/{task}/train.npz"),
                    "--val-cache", str(output / f"cache/{task}/val.npz"),
                    "--official-checkpoint",
                    str(output / "pretrained/lewm-reacher/weights.pt"),
                    "--output-dir", str(directory),
                    "--history-len", str(history),
                    "--epochs", "60", "--batch-size", "256",
                    "--lr", "0.0001", "--weight-decay", "0.001",
                    "--seed", str(seed), "--device", "cuda",
                )
                jobs.append(Job(f"{task}_h{history}_s{seed}", command,
                                directory / "metrics.json"))
    return jobs


def rollout_jobs(output: Path) -> list[Job]:
    jobs = []
    for task in TASKS:
        for objective in ("one_step", "overshoot_8"):
            for seed in range(3):
                directory = output / "rollout" / task / objective / f"s{seed}"
                command = (
                    PYTHON, "scripts/train_official_rollout.py",
                    "--task", task, "--objective", objective,
                    "--seed", str(seed),
                    "--cache-root", str(output / "cache"),
                    "--weights", str(output / "pretrained/lewm-reacher/weights.pt"),
                    "--output", str(output / "rollout"),
                    "--epochs", "60", "--batch-size", "64",
                    "--lr", "0.0001", "--weight-decay", "0.001",
                    "--device", "cuda",
                )
                jobs.append(Job(f"{task}_{objective}_s{seed}", command,
                                directory / "metrics.json"))
    return jobs


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave", required=True,
                        choices=("frozen", "context", "rollout"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/paper_a_expansion"))
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument(
        "--shard-count", type=int, default=1,
        help="Deterministically split the canonical job list into this many shards.")
    parser.add_argument(
        "--shard-index", type=int, default=0,
        help="Zero-based shard to run; applied before completed jobs are skipped.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    if args.shard_count < 1:
        raise ValueError("--shard-count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must lie in [0, shard-count)")
    builders = {"frozen": frozen_jobs, "context": context_jobs,
                "rollout": rollout_jobs}
    canonical = builders[args.wave](args.output)
    assigned = [job for index, job in enumerate(canonical)
                if index % args.shard_count == args.shard_index]
    queue = [job for job in assigned if not job.done_file.is_file()]
    logs = args.output / "logs" / args.wave
    logs.mkdir(parents=True, exist_ok=True)
    print(f"[paper-a-launch] wave={args.wave} pending={len(queue)} "
          f"parallel={args.jobs} shard={args.shard_index}/{args.shard_count}",
          flush=True)
    environment = dict(os.environ)
    environment.setdefault("PYTHONHASHSEED", "0")
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # These models are GPU-resident and each launcher intentionally runs
    # several independent seeds at once.  Unbounded BLAS/OpenMP pools cause
    # severe CPU oversubscription without changing the numerical workload.
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                     "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment.setdefault(variable, "1")
    running: list[tuple[subprocess.Popen, Job, object]] = []
    completed = failed = 0
    while queue or running:
        while queue and len(running) < args.jobs:
            job = queue.pop(0)
            stream = (logs / f"{job.name}.log").open("w")
            process = subprocess.Popen(
                job.command, cwd=ROOT, env=environment,
                stdout=stream, stderr=subprocess.STDOUT)
            running.append((process, job, stream))
            print(f"[paper-a-launch] start {job.name}", flush=True)
        time.sleep(2)
        active = []
        for process, job, stream in running:
            code = process.poll()
            if code is None:
                active.append((process, job, stream))
                continue
            stream.close()
            if code == 0 and job.done_file.is_file():
                completed += 1
                print(f"[paper-a-launch] done {job.name}", flush=True)
            else:
                failed += 1
                print(f"[paper-a-launch] FAIL {job.name} exit={code} "
                      f"log={logs / (job.name + '.log')}", flush=True)
        running = active
        print(f"[paper-a-launch] progress ok={completed} fail={failed} "
              f"queued={len(queue)} active={len(running)}", flush=True)
    if failed:
        raise SystemExit(f"{failed} jobs failed")


if __name__ == "__main__":
    main()
