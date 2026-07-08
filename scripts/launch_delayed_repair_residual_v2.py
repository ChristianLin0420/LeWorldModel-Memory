#!/usr/bin/env python3
"""Preview or explicitly run the development-gated delayed repair V2 study."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.delayed_repair_residual_v2_spec import (  # noqa: E402
    ARMS,
    CONDITIONS,
    DEFAULT_LOCK,
    DEFAULT_SPEC,
    SEEDS,
    TASKS,
    development_receipt_path,
    load_locked_spec,
    repair_directory,
    require_development_health,
    resolve_path,
    validate_device,
)


PYTHON = sys.executable
WAVES = ("development-health", "formal-repair", "aggregate")
ALL_WAVES = (*WAVES, "all")


@dataclass(frozen=True)
class Job:
    name: str
    command: tuple[str, ...]
    done_file: Path
    device: str | None


def parse_gpu_ids(raw: str) -> tuple[int, ...]:
    tokens = [token.strip() for token in raw.split(",")]
    if not tokens or any(not token for token in tokens):
        raise ValueError("--gpus must be a comma-separated list such as 1,2")
    result = []
    for token in tokens:
        if token.startswith("cuda:"):
            token = token.split(":", 1)[1]
        if not token.isdigit():
            raise ValueError(f"invalid GPU identifier {token!r}")
        gpu_id = int(token)
        validate_device(f"cuda:{gpu_id}")
        if gpu_id in result:
            raise ValueError(f"duplicate GPU identifier {gpu_id}")
        result.append(gpu_id)
    return tuple(result)


def build_wave_jobs(
        spec: Mapping, wave: str, gpu_ids: Sequence[int],
        spec_path: Path = DEFAULT_SPEC, lock_path: Path = DEFAULT_LOCK,
        ) -> list[Job]:
    if wave not in WAVES:
        raise ValueError(f"unknown residual-repair V2 wave {wave!r}")
    if not gpu_ids:
        raise ValueError("at least one allowed GPU is required")
    devices = tuple(validate_device(f"cuda:{int(gpu)}") for gpu in gpu_ids)
    common = ("--spec", str(spec_path), "--lock", str(lock_path))
    jobs = []
    if wave == "development-health":
        for task in TASKS:
            jobs.append(Job(
                f"development-health-{task}",
                (PYTHON,
                 "scripts/audit_delayed_repair_residual_v2_development.py",
                 "--task", task, *common, "--execute"),
                development_receipt_path(spec, task), None))
    elif wave == "formal-repair":
        base_cell = 0
        for task in TASKS:
            for arm in ARMS:
                for seed in SEEDS:
                    device = devices[base_cell % len(devices)]
                    for condition in CONDITIONS:
                        jobs.append(Job(
                            (f"{task}-{arm}-checkpoint-seed-{seed}-"
                             f"{condition}"),
                            (PYTHON,
                             "scripts/train_delayed_repair_residual_v2.py",
                             "--task", task, "--arm", arm,
                             "--seed", str(seed), "--condition", condition,
                             "--device", device, *common, "--execute"),
                            repair_directory(
                                spec, task, arm, seed, condition)
                            / "manifest.json",
                            device))
                    base_cell += 1
    else:
        jobs.append(Job(
            "aggregate-label-free-residual-diagnostics",
            (PYTHON, "scripts/aggregate_delayed_repair_residual_v2.py",
             *common, "--execute"),
            resolve_path(spec["output"]["summary"]), None))
    names = [job.name for job in jobs]
    outputs = [job.done_file for job in jobs]
    if len(names) != len(set(names)) or len(outputs) != len(set(outputs)):
        raise RuntimeError(f"duplicate delayed repair V2 job in {wave}")
    for job in jobs:
        if job.device is not None:
            validate_device(job.device)
    return jobs


def build_plan(
        spec: Mapping, wave: str, gpu_ids: Sequence[int],
        spec_path: Path = DEFAULT_SPEC, lock_path: Path = DEFAULT_LOCK,
        ) -> list[tuple[str, list[Job]]]:
    if wave not in ALL_WAVES:
        raise ValueError(f"unknown residual-repair V2 wave {wave!r}")
    selected = WAVES if wave == "all" else (wave,)
    return [(name, build_wave_jobs(
        spec, name, gpu_ids, spec_path, lock_path)) for name in selected]


def preview_lines(plan: Sequence[tuple[str, Sequence[Job]]]) -> list[str]:
    lines = []
    for wave, jobs in plan:
        for job in jobs:
            status = "complete" if job.done_file.is_file() else "pending"
            lines.append(
                f"{wave}\t{status}\t{job.name}\t{shlex.join(job.command)}")
    return lines


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave", required=True, choices=ALL_WAVES)
    parser.add_argument("--gpus", default="1,2")
    parser.add_argument("--cpu-jobs", type=int, default=2)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.setdefault("PYTHONHASHSEED", "0")
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for variable in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment.setdefault(variable, "1")
    return environment


def _next_startable(queue: Sequence[Job], running: Sequence[tuple],
                    cpu_jobs: int) -> int | None:
    busy = {job.device for _, job, _ in running if job.device is not None}
    active_cpu = sum(job.device is None for _, job, _ in running)
    for index, job in enumerate(queue):
        if job.device is None and active_cpu < cpu_jobs:
            return index
        if job.device is not None and job.device not in busy:
            return index
    return None


def _execute_wave(spec: Mapping, wave: str, jobs: Sequence[Job],
                  cpu_jobs: int) -> None:
    if wave in ("formal-repair", "aggregate"):
        for task in TASKS:
            require_development_health(spec, task)
    pending = [job for job in jobs if not job.done_file.is_file()]
    if not pending:
        print(f"[delayed-residual-v2-launch] {wave} already complete", flush=True)
        return
    logs = resolve_path(spec["output"]["logs"]) / wave
    logs.mkdir(parents=True, exist_ok=True)
    queue = list(pending)
    running: list[tuple[subprocess.Popen, Job, object]] = []
    completed = failed = 0
    while queue or running:
        while True:
            index = _next_startable(queue, running, cpu_jobs)
            if index is None:
                break
            job = queue.pop(index)
            stream = (logs / f"{job.name}.log").open("x")
            try:
                process = subprocess.Popen(
                    job.command, cwd=ROOT, env=_environment(),
                    stdout=stream, stderr=subprocess.STDOUT)
            except BaseException:
                stream.close()
                raise
            running.append((process, job, stream))
            print(f"[delayed-residual-v2-launch] start {job.name} "
                  f"device={job.device or 'cpu'}", flush=True)
        if not running and queue:
            raise RuntimeError("residual-repair V2 scheduler stalled")
        time.sleep(1)
        active = []
        for process, job, stream in running:
            code = process.poll()
            if code is None:
                active.append((process, job, stream))
                continue
            stream.close()
            if code == 0 and job.done_file.is_file():
                completed += 1
                print(f"[delayed-residual-v2-launch] done {job.name}", flush=True)
            else:
                failed += 1
                print(f"[delayed-residual-v2-launch] FAIL {job.name} "
                      f"exit={code}", flush=True)
        running = active
    if failed:
        raise SystemExit(
            f"{failed} jobs failed in {wave}; downstream V2 waves blocked")
    print(f"[delayed-residual-v2-launch] wave={wave} "
          f"complete={completed}", flush=True)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.cpu_jobs < 1:
        raise ValueError("--cpu-jobs must be positive")
    gpu_ids = parse_gpu_ids(args.gpus)
    spec_path, lock_path = args.spec.resolve(), args.lock.resolve()
    spec = load_locked_spec(spec_path, lock_path)
    plan = build_plan(spec, args.wave, gpu_ids, spec_path, lock_path)
    canonical = sum(len(jobs) for _, jobs in plan)
    pending = sum(
        not job.done_file.is_file() for _, jobs in plan for job in jobs)
    print(f"[delayed-residual-v2-launch] wave={args.wave} "
          f"canonical={canonical} pending={pending} "
          f"gpus={','.join(map(str, gpu_ids))} execute={args.execute}",
          flush=True)
    if not args.execute:
        for line in preview_lines(plan):
            print(line)
        return
    for wave, jobs in plan:
        _execute_wave(spec, wave, jobs, args.cpu_jobs)


if __name__ == "__main__":
    main()
