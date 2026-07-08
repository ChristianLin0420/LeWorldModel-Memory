#!/usr/bin/env python3
"""Preview or explicitly run the immutable delayed-goal use waves.

Preview is read-only.  Execution requires ``--execute`` and is restricted to
CUDA devices 1 and 2 before any output or log directory can be created.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.paper_a_delayed_goal_spec import (
    DEFAULT_SPEC,
    REPAIR_ARMS,
    REPAIR_CONDITIONS,
    SEEDS,
    TASKS,
    evaluation_directory,
    load_locked_spec,
    repair_directory,
    resolve_path,
    task_slug,
    validate_device,
)


PYTHON = sys.executable
WAVES = ("repair", "evaluate", "aggregate")


@dataclass(frozen=True)
class Job:
    name: str
    command: tuple[str, ...]
    done_file: Path


def build_jobs(spec: Mapping, wave: str, device: str,
               spec_path: Path = DEFAULT_SPEC) -> list[Job]:
    if wave not in WAVES:
        raise ValueError(f"unknown delayed-goal wave {wave!r}")
    validate_device(spec, device)
    jobs: list[Job] = []
    if wave == "repair":
        for task in TASKS:
            for arm in REPAIR_ARMS:
                for seed in SEEDS:
                    for condition in REPAIR_CONDITIONS:
                        jobs.append(Job(
                            (f"{task_slug(spec, task)}_{arm.replace('_', '-')}"
                             f"_checkpoint-seed-{seed}_{condition.replace('_', '-') }"),
                            (PYTHON,
                             "scripts/train_paper_a_delayed_goal_repair.py",
                             "--spec", str(spec_path), "--task", task,
                             "--arm", arm, "--seed", str(seed),
                             "--condition", condition, "--device", device,
                             "--execute"),
                            repair_directory(
                                spec, task, arm, seed, condition)
                            / "metrics.json",
                        ))
    elif wave == "evaluate":
        for task in TASKS:
            for seed in SEEDS:
                jobs.append(Job(
                    f"{task_slug(spec, task)}_checkpoint-seed-{seed}",
                    (PYTHON, "scripts/evaluate_paper_a_delayed_goal_use.py",
                     "--spec", str(spec_path), "--task", task,
                     "--seed", str(seed), "--device", device, "--execute"),
                    evaluation_directory(spec, task, seed) / "metrics.json",
                ))
    else:
        jobs.append(Job(
            "crossed-paired-bootstrap-summary",
            (PYTHON, "scripts/aggregate_paper_a_delayed_goal_use.py",
             "--spec", str(spec_path), "--execute"),
            resolve_path(spec["output"]["summary"]),
        ))
    names = [job.name for job in jobs]
    outputs = [job.done_file for job in jobs]
    if len(names) != len(set(names)) or len(outputs) != len(set(outputs)):
        raise RuntimeError("delayed-goal job grid contains duplicate cells")
    parent = resolve_path(spec["parent"]["root"])
    for output in outputs:
        if output == parent or parent in output.parents:
            raise RuntimeError(f"job would modify parent artifact: {output}")
    return jobs


def select_shard(jobs: list[Job], wave: str, count: int,
                 index: int) -> list[Job]:
    if count < 1 or not 0 <= index < count:
        raise ValueError("invalid shard selection")
    divisor = len(REPAIR_CONDITIONS) if wave == "repair" else 1
    return [job for job_index, job in enumerate(jobs)
            if (job_index // divisor) % count == index]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--wave", required=True, choices=WAVES)
    parser.add_argument("--device", default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    spec = load_locked_spec(args.spec)
    device = args.device or spec["execution"]["default_device"]
    validate_device(spec, device)
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard selection")
    canonical = build_jobs(spec, args.wave, device, args.spec.resolve())
    # Keep objective-off/cue-repair twins on the same device when the repair
    # wave is sharded across CUDA 1 and 2.
    assigned = select_shard(
        canonical, args.wave, args.shard_count, args.shard_index)
    pending = [job for job in assigned if not job.done_file.is_file()]
    print(
        f"[delayed-goal-launch] wave={args.wave} canonical={len(canonical)} "
        f"assigned={len(assigned)} pending={len(pending)} device={device} "
        f"execute={args.execute}", flush=True)
    if not args.execute:
        for job in pending:
            print(f"{job.name}\t{shlex.join(job.command)}")
        return

    logs = resolve_path(spec["output"]["logs"]) / args.wave
    logs.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.setdefault("PYTHONHASHSEED", "0")
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                     "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment.setdefault(variable, "1")
    queue = list(pending)
    running: list[tuple[subprocess.Popen, Job, object]] = []
    completed = failed = 0
    while queue or running:
        while queue and len(running) < args.jobs:
            job = queue.pop(0)
            stream = (logs / f"{job.name}.log").open("x")
            process = subprocess.Popen(
                job.command, cwd=ROOT, env=environment,
                stdout=stream, stderr=subprocess.STDOUT)
            running.append((process, job, stream))
            print(f"[delayed-goal-launch] start {job.name}", flush=True)
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
                print(f"[delayed-goal-launch] done {job.name}", flush=True)
            else:
                failed += 1
                print(f"[delayed-goal-launch] FAIL {job.name} exit={code}",
                      flush=True)
        running = active
    if failed:
        raise SystemExit(f"{failed} delayed-goal jobs failed")
    print(f"[delayed-goal-launch] complete={completed}", flush=True)


if __name__ == "__main__":
    main()
