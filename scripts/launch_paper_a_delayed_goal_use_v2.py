#!/usr/bin/env python3
"""Preview or explicitly run the controller-locked delayed-goal V2 waves."""

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

from scripts.paper_a_delayed_goal_v2_spec import (
    DEFAULT_SPEC,
    SEEDS,
    TASKS,
    controller_lock_paths,
    evaluation_directory,
    load_controller_lock,
    load_locked_spec,
    resolve_path,
    task_slug,
    validate_device,
)


PYTHON = sys.executable
WAVES = ("controller-select", "evaluate", "aggregate")


@dataclass(frozen=True)
class Job:
    name: str
    command: tuple[str, ...]
    done_file: Path


def build_jobs(spec: Mapping, wave: str, device: str,
               spec_path: Path = DEFAULT_SPEC) -> list[Job]:
    if wave not in WAVES:
        raise ValueError(f"unknown V2 wave {wave!r}")
    validate_device(spec, device)
    if wave == "controller-select":
        lock, _ = controller_lock_paths(spec)
        jobs = [Job(
            "training-bank-controller-health-selection",
            (PYTHON, "scripts/select_paper_a_delayed_goal_v2_controller.py",
             "--spec", str(spec_path), "--execute"),
            lock,
        )]
    elif wave == "evaluate":
        jobs = [
            Job(
                f"{task_slug(spec, task)}_checkpoint-seed-{seed}",
                (PYTHON, "scripts/evaluate_paper_a_delayed_goal_use_v2.py",
                 "--spec", str(spec_path), "--task", task,
                 "--seed", str(seed), "--device", device, "--execute"),
                evaluation_directory(spec, task, seed) / "metrics.json",
            )
            for task in TASKS for seed in SEEDS
        ]
    else:
        jobs = [Job(
            "crossed-paired-bootstrap-summary",
            (PYTHON, "scripts/aggregate_paper_a_delayed_goal_use_v2.py",
             "--spec", str(spec_path), "--execute"),
            resolve_path(spec["output"]["summary"]),
        )]
    if len({job.name for job in jobs}) != len(jobs) \
            or len({job.done_file for job in jobs}) != len(jobs):
        raise RuntimeError("V2 job grid contains duplicate cells")
    v1_root = resolve_path(spec["v1"]["output_root"])
    v2_root = resolve_path(spec["output"]["root"])
    for job in jobs:
        if v2_root not in job.done_file.parents \
                or v1_root in job.done_file.parents:
            raise RuntimeError(f"V2 job output is not isolated: {job.done_file}")
        if "train_paper_a_delayed_goal_repair.py" in job.command:
            raise RuntimeError("V2 must never schedule repair retraining")
    return jobs


def select_shard(jobs: list[Job], count: int, index: int) -> list[Job]:
    if count < 1 or not 0 <= index < count:
        raise ValueError("invalid shard selection")
    return [job for job_index, job in enumerate(jobs)
            if job_index % count == index]


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
    canonical = build_jobs(spec, args.wave, device, args.spec.resolve())
    assigned = select_shard(
        canonical, args.shard_count, args.shard_index)
    pending = [job for job in assigned if not job.done_file.is_file()]
    print(
        f"[delayed-goal-v2-launch] wave={args.wave} "
        f"canonical={len(canonical)} assigned={len(assigned)} "
        f"pending={len(pending)} device={device} execute={args.execute}",
        flush=True)
    if not args.execute:
        for job in pending:
            print(f"{job.name}\t{shlex.join(job.command)}")
        return
    if args.wave in ("evaluate", "aggregate"):
        # This occurs before log/output creation and is the hard separation
        # between training-bank controller development and validation.
        load_controller_lock(spec)

    logs = resolve_path(spec["output"]["logs"]) / args.wave
    logs.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.setdefault("PYTHONHASHSEED", "0")
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                     "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment.setdefault(variable, "1")
    if args.wave == "controller-select":
        environment["MUJOCO_GL"] = spec["execution"][
            "controller_selection_gl_backend"]
        environment["CUDA_VISIBLE_DEVICES"] = ""
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
            print(f"[delayed-goal-v2-launch] start {job.name}", flush=True)
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
                print(f"[delayed-goal-v2-launch] done {job.name}", flush=True)
            else:
                failed += 1
                print(f"[delayed-goal-v2-launch] FAIL {job.name} exit={code}",
                      flush=True)
        running = active
    if failed:
        raise SystemExit(f"{failed} delayed-goal V2 jobs failed")
    print(f"[delayed-goal-v2-launch] complete={completed}", flush=True)


if __name__ == "__main__":
    main()
