#!/usr/bin/env python3
"""Preview or explicitly execute the development-gated shell-game V3 plan."""

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

from lewm.official_tasks.shell_game_capacity import CAPACITY_STAGES  # noqa: E402
from lewm.official_tasks.shell_game_pipeline_v3 import (  # noqa: E402
    base_path_v3,
    cache_manifest_path_v3,
    carrier_directory_v3,
    development_manifest_path_v3,
    log_root_v3,
    require_all_selected_salience_v3,
    stage_path_v3,
)
from lewm.official_tasks.shell_game_spec_v3 import (  # noqa: E402
    DEFAULT_LOCK_V3,
    DEFAULT_SPEC_V3,
    load_locked_spec_v3,
    validate_device_v3,
)


PYTHON = sys.executable
WAVES = (
    "development-base",
    "development-stages",
    "development-salience",
    "formal-base",
    "formal-stages",
    "formal-cache",
    "carriers",
)
ALL_WAVES = (*WAVES, "all")
FORMAL_WAVES = {"formal-base", "formal-stages", "formal-cache", "carriers"}


@dataclass(frozen=True)
class JobV3:
    name: str
    command: tuple[str, ...]
    done_file: Path
    device: str | None


def parse_gpu_ids_v3(raw: str) -> tuple[int, ...]:
    tokens = [token.strip() for token in raw.split(",")]
    if not tokens or any(not token for token in tokens):
        raise ValueError("--gpus must be a comma-separated list such as 1,2")
    result: list[int] = []
    for token in tokens:
        if token.startswith("cuda:"):
            token = token.split(":", 1)[1]
        if not token.isdigit():
            raise ValueError(f"invalid GPU identifier {token!r}")
        gpu_id = int(token)
        validate_device_v3(f"cuda:{gpu_id}")
        if gpu_id in result:
            raise ValueError(f"duplicate GPU identifier {gpu_id}")
        result.append(gpu_id)
    return tuple(result)


def _sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def build_wave_jobs_v3(
        spec: Mapping, wave: str, gpu_ids: Sequence[int],
        spec_path: Path = DEFAULT_SPEC_V3,
        lock_path: Path = DEFAULT_LOCK_V3,
        ) -> list[JobV3]:
    """Build one immutable V3 wave without touching output directories."""

    if wave not in WAVES:
        raise ValueError(f"unknown shell-game V3 wave {wave!r}")
    if not gpu_ids:
        raise ValueError("at least one allowed GPU is required")
    devices = tuple(validate_device_v3(f"cuda:{int(gpu_id)}")
                    for gpu_id in gpu_ids)
    stages = tuple(stage.key for stage in CAPACITY_STAGES)
    common = ("--spec", str(spec_path), "--lock", str(lock_path))
    jobs: list[JobV3] = []

    if wave == "development-base":
        destination = base_path_v3(spec, "development")
        jobs.append(JobV3(
            "development-base",
            (PYTHON, "scripts/collect_official_shell_game_base_v3.py",
             "--split", "development", "--device", devices[0], *common),
            _sidecar(destination), devices[0]))
    elif wave == "development-stages":
        for stage in stages:
            destination = stage_path_v3(spec, stage, "development")
            jobs.append(JobV3(
                f"development-stage-{stage}",
                (PYTHON, "scripts/prepare_official_shell_game_stage_v3.py",
                 "--stage", stage, "--split", "development", *common),
                _sidecar(destination), None))
    elif wave == "development-salience":
        for index, stage in enumerate(stages):
            device = devices[index % len(devices)]
            jobs.append(JobV3(
                f"development-salience-{stage}",
                (PYTHON,
                 "scripts/cache_official_shell_game_development_v3.py",
                 "--stage", stage, "--device", device, *common),
                development_manifest_path_v3(spec, stage), device))
    elif wave == "formal-base":
        for index, split in enumerate(("train", "validation")):
            device = devices[index % len(devices)]
            destination = base_path_v3(spec, split)
            jobs.append(JobV3(
                f"formal-base-{split}",
                (PYTHON, "scripts/collect_official_shell_game_base_v3.py",
                 "--split", split, "--device", device, *common),
                _sidecar(destination), device))
    elif wave == "formal-stages":
        for stage in stages:
            for split in ("train", "validation"):
                destination = stage_path_v3(spec, stage, split)
                jobs.append(JobV3(
                    f"formal-stage-{stage}-{split}",
                    (PYTHON,
                     "scripts/prepare_official_shell_game_stage_v3.py",
                     "--stage", stage, "--split", split, *common),
                    _sidecar(destination), None))
    elif wave == "formal-cache":
        for index, stage in enumerate(stages):
            device = devices[index % len(devices)]
            jobs.append(JobV3(
                f"formal-cache-{stage}",
                (PYTHON, "scripts/cache_official_shell_game_capacity_v3.py",
                 "--stage", stage, "--device", device, *common),
                cache_manifest_path_v3(spec, stage), device))
    else:
        index = 0
        training = spec["carrier_training"]
        for stage in stages:
            for arm in training["arms"]:
                for seed in training["seeds"]:
                    device = devices[index % len(devices)]
                    jobs.append(JobV3(
                        f"carrier-{stage}-{arm}-seed-{int(seed)}",
                        (PYTHON,
                         "scripts/train_official_shell_game_capacity_v3.py",
                         "--stage", stage, "--arm", arm,
                         "--seed", str(int(seed)),
                         "--device", device, *common),
                        carrier_directory_v3(
                            spec, stage, arm, int(seed)) / "manifest.json",
                        device))
                    index += 1

    names = [job.name for job in jobs]
    destinations = [job.done_file for job in jobs]
    if len(names) != len(set(names)) \
            or len(destinations) != len(set(destinations)):
        raise RuntimeError(f"duplicate V3 job cell in {wave}")
    for job in jobs:
        if job.device is not None:
            validate_device_v3(job.device)
    return jobs


def build_plan_v3(
        spec: Mapping, wave: str, gpu_ids: Sequence[int],
        spec_path: Path = DEFAULT_SPEC_V3,
        lock_path: Path = DEFAULT_LOCK_V3,
        ) -> list[tuple[str, list[JobV3]]]:
    if wave not in ALL_WAVES:
        raise ValueError(f"unknown shell-game V3 wave {wave!r}")
    selected = WAVES if wave == "all" else (wave,)
    return [(name, build_wave_jobs_v3(
        spec, name, gpu_ids, spec_path, lock_path)) for name in selected]


def preview_lines_v3(
        plan: Sequence[tuple[str, Sequence[JobV3]]]) -> list[str]:
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
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_V3)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK_V3)
    parser.add_argument(
        "--execute", action="store_true",
        help="execute V3; without this flag only print the immutable plan")
    return parser.parse_args(argv)


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.setdefault("PYTHONHASHSEED", "0")
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    environment.setdefault("MUJOCO_GL", "egl")
    for variable in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment.setdefault(variable, "1")
    return environment


def _next_startable(queue: Sequence[JobV3], running: Sequence[tuple],
                    cpu_jobs: int) -> int | None:
    busy_devices = {
        job.device for _, job, _ in running if job.device is not None}
    active_cpu = sum(job.device is None for _, job, _ in running)
    for index, job in enumerate(queue):
        if job.device is None and active_cpu < cpu_jobs:
            return index
        if job.device is not None and job.device not in busy_devices:
            return index
    return None


def _execute_wave(spec: Mapping, wave: str, jobs: Sequence[JobV3],
                  cpu_jobs: int) -> None:
    if wave in FORMAL_WAVES:
        require_all_selected_salience_v3(spec)
    pending = [job for job in jobs if not job.done_file.is_file()]
    if not pending:
        print(f"[shell-game-v3-launch] wave={wave} already complete", flush=True)
        return
    logs = log_root_v3(spec) / wave
    logs.mkdir(parents=True, exist_ok=True)
    queue = list(pending)
    running: list[tuple[subprocess.Popen, JobV3, object]] = []
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
            print(f"[shell-game-v3-launch] start {job.name} "
                  f"device={job.device or 'cpu'}", flush=True)
        if not running and queue:
            raise RuntimeError("V3 scheduler cannot start a pending job")
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
                print(f"[shell-game-v3-launch] done {job.name}", flush=True)
            else:
                failed += 1
                print(f"[shell-game-v3-launch] FAIL {job.name} exit={code} "
                      f"log={logs / (job.name + '.log')}", flush=True)
        running = active
    if failed:
        raise SystemExit(
            f"{failed} V3 jobs failed in {wave}; downstream waves blocked")
    print(f"[shell-game-v3-launch] wave={wave} complete={completed}", flush=True)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.cpu_jobs < 1:
        raise ValueError("--cpu-jobs must be positive")
    gpu_ids = parse_gpu_ids_v3(args.gpus)
    spec_path, lock_path = args.spec.resolve(), args.lock.resolve()
    spec = load_locked_spec_v3(spec_path, lock_path)
    plan = build_plan_v3(spec, args.wave, gpu_ids, spec_path, lock_path)
    canonical = sum(len(jobs) for _, jobs in plan)
    pending = sum(
        not job.done_file.is_file() for _, jobs in plan for job in jobs)
    print(f"[shell-game-v3-launch] wave={args.wave} canonical={canonical} "
          f"pending={pending} gpus={','.join(map(str, gpu_ids))} "
          f"execute={args.execute}", flush=True)
    if not args.execute:
        for line in preview_lines_v3(plan):
            print(line)
        return
    for wave, jobs in plan:
        _execute_wave(spec, wave, jobs, args.cpu_jobs)


if __name__ == "__main__":
    main()
