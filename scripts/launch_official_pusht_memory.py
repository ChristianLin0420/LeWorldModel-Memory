#!/usr/bin/env python3
"""Preview or explicitly launch the locked official-PushT memory pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.official_tasks.pusht_pipeline import (  # noqa: E402
    pusht_base_manifest_path,
    pusht_carrier_directory,
    pusht_log_root,
    pusht_task_manifest_path,
)
from lewm.official_tasks.pusht_spec import (  # noqa: E402
    DEFAULT_PUSHT_LOCK,
    DEFAULT_PUSHT_SPEC,
    load_locked_pusht_spec,
    validate_pusht_device,
)


PYTHON = sys.executable
PUSHT_WAVES = ("base-cache", "task-cache", "carriers")
PUSHT_ALL_WAVES = (*PUSHT_WAVES, "all")


@dataclass(frozen=True)
class PushTJob:
    name: str
    command: tuple[str, ...]
    done_file: Path
    device: str


def parse_pusht_gpu_ids(value: str) -> tuple[int, ...]:
    tokens = [token.strip() for token in value.split(",")]
    if not tokens or any(not token for token in tokens):
        raise ValueError("--gpus must be a comma-separated list such as 1,2")
    result = []
    for token in tokens:
        if token.startswith("cuda:"):
            token = token.split(":", 1)[1]
        if not token.isdigit():
            raise ValueError(f"invalid GPU identifier {token!r}")
        gpu = int(token)
        validate_pusht_device(f"cuda:{gpu}")
        if gpu in result:
            raise ValueError(f"duplicate GPU identifier {gpu}")
        result.append(gpu)
    return tuple(result)


def build_pusht_wave_jobs(
        spec: Mapping, wave: str, gpu_ids: Sequence[int],
        spec_path: Path = DEFAULT_PUSHT_SPEC,
        lock_path: Path = DEFAULT_PUSHT_LOCK) -> list[PushTJob]:
    if wave not in PUSHT_WAVES:
        raise ValueError(f"unknown formal PushT wave {wave!r}")
    if not gpu_ids:
        raise ValueError("at least one allowed GPU is required")
    devices = tuple(validate_pusht_device(f"cuda:{int(gpu)}")
                    for gpu in gpu_ids)
    common = ("--spec", str(spec_path), "--lock", str(lock_path), "--execute")
    tasks = [task["key"] for task in spec["semantic_tasks"]]
    jobs: list[PushTJob] = []
    if wave == "base-cache":
        jobs.append(PushTJob(
            name="base-cache",
            command=(
                PYTHON, "scripts/cache_official_pusht_memory.py",
                "--phase", "base", "--device", devices[0], *common),
            done_file=pusht_base_manifest_path(spec),
            device=devices[0],
        ))
    elif wave == "task-cache":
        for index, task in enumerate(tasks):
            device = devices[index % len(devices)]
            jobs.append(PushTJob(
                name=f"task-cache-{task}",
                command=(
                    PYTHON, "scripts/cache_official_pusht_memory.py",
                    "--phase", "task", "--task", task,
                    "--device", device, *common),
                done_file=pusht_task_manifest_path(spec, task),
                device=device,
            ))
    else:
        index = 0
        for task in tasks:
            for arm in spec["carrier_training"]["arms"]:
                for seed in spec["carrier_training"]["seeds"]:
                    device = devices[index % len(devices)]
                    jobs.append(PushTJob(
                        name=f"carrier-{task}-{arm}-seed-{seed}",
                        command=(
                            PYTHON, "scripts/train_official_pusht_carrier.py",
                            "--task", task, "--arm", arm,
                            "--seed", str(seed), "--device", device, *common),
                        done_file=(pusht_carrier_directory(
                            spec, task, arm, seed) / "manifest.json"),
                        device=device,
                    ))
                    index += 1
    if len({job.name for job in jobs}) != len(jobs) \
            or len({job.done_file for job in jobs}) != len(jobs):
        raise RuntimeError(f"duplicate formal PushT job in wave {wave}")
    return jobs


def build_pusht_plan(
        spec: Mapping, wave: str, gpu_ids: Sequence[int],
        spec_path: Path = DEFAULT_PUSHT_SPEC,
        lock_path: Path = DEFAULT_PUSHT_LOCK,
        ) -> list[tuple[str, list[PushTJob]]]:
    if wave not in PUSHT_ALL_WAVES:
        raise ValueError(f"unknown formal PushT wave {wave!r}")
    selected = PUSHT_WAVES if wave == "all" else (wave,)
    return [(name, build_pusht_wave_jobs(
        spec, name, gpu_ids, spec_path, lock_path)) for name in selected]


def preview_pusht_plan(
        plan: Sequence[tuple[str, Sequence[PushTJob]]]) -> list[str]:
    return [
        f"{wave}\t{'complete' if job.done_file.is_file() else 'pending'}\t"
        f"{job.name}\t{shlex.join(job.command)}"
        for wave, jobs in plan for job in jobs
    ]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave", required=True, choices=PUSHT_ALL_WAVES)
    parser.add_argument("--gpus", default="1,2")
    parser.add_argument("--spec", type=Path, default=DEFAULT_PUSHT_SPEC)
    parser.add_argument("--lock", type=Path, default=DEFAULT_PUSHT_LOCK)
    parser.add_argument(
        "--execute", action="store_true",
        help="without this flag the immutable job grid is only printed")
    return parser.parse_args(argv)


def _environment() -> dict[str, str]:
    value = dict(os.environ)
    value.setdefault("PYTHONHASHSEED", "0")
    value.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        value.setdefault(name, "1")
    return value


def _execute_wave(spec: Mapping, wave: str,
                  jobs: Sequence[PushTJob]) -> None:
    queue = [job for job in jobs if not job.done_file.is_file()]
    if not queue:
        print(f"[pusht-launch] wave={wave} already complete", flush=True)
        return
    logs = pusht_log_root(spec) / wave
    logs.mkdir(parents=True, exist_ok=True)
    running: list[tuple[subprocess.Popen, PushTJob, object]] = []
    failed = 0
    while queue or running:
        busy = {job.device for _, job, _ in running}
        for job in list(queue):
            if job.device in busy:
                continue
            stream = (logs / f"{job.name}.log").open("x")
            process = subprocess.Popen(
                job.command, cwd=ROOT, env=_environment(),
                stdout=stream, stderr=subprocess.STDOUT)
            running.append((process, job, stream))
            queue.remove(job)
            busy.add(job.device)
        if not running and queue:
            raise RuntimeError("formal PushT scheduler cannot start pending jobs")
        time.sleep(1)
        active = []
        for process, job, stream in running:
            code = process.poll()
            if code is None:
                active.append((process, job, stream))
                continue
            stream.close()
            if code != 0 or not job.done_file.is_file():
                failed += 1
                print(f"[pusht-launch] FAIL {job.name} exit={code}", flush=True)
            else:
                print(f"[pusht-launch] done {job.name}", flush=True)
        running = active
    if failed:
        raise SystemExit(
            f"{failed} jobs failed in {wave}; later waves were not started")


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    gpu_ids = parse_pusht_gpu_ids(args.gpus)
    spec = load_locked_pusht_spec(args.spec, args.lock)
    plan = build_pusht_plan(
        spec, args.wave, gpu_ids, args.spec.resolve(), args.lock.resolve())
    print(f"[pusht-launch] wave={args.wave} jobs="
          f"{sum(len(jobs) for _, jobs in plan)} execute={args.execute}")
    if not args.execute:
        for line in preview_pusht_plan(plan):
            print(line)
        return
    for wave, jobs in plan:
        _execute_wave(spec, wave, jobs)


if __name__ == "__main__":
    main()
