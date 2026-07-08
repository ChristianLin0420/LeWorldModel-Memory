#!/usr/bin/env python3
"""Preview or explicitly execute the locked semantic shell-game pipeline.

The launcher is read-only unless ``--execute`` is present.  GPU-backed EGL
collection, frozen encoding, and carrier fitting are pinned directly to the
absolute devices ``cuda:1`` and ``cuda:2``; devices 0 and 3 are never
accepted or remapped through ``CUDA_VISIBLE_DEVICES``.
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
from typing import Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.official_tasks.shell_game_capacity import (  # noqa: E402
    CAPACITY_STAGES,
)
from lewm.official_tasks.shell_game_pipeline import (  # noqa: E402
    SPLITS,
    base_path,
    cache_manifest_path,
    carrier_directory,
    log_root,
    stage_path,
)
from lewm.official_tasks.shell_game_spec import (  # noqa: E402
    DEFAULT_LOCK,
    DEFAULT_SPEC,
    load_locked_spec,
    validate_device,
)


PYTHON = sys.executable
WAVES = ("base", "stages", "frozen-cache", "carriers")
ALL_WAVES = (*WAVES, "all")


@dataclass(frozen=True)
class Job:
    """One immutable process cell and its completion sentinel."""

    name: str
    command: tuple[str, ...]
    done_file: Path
    device: str | None


def parse_gpu_ids(raw: str) -> tuple[int, ...]:
    """Parse an absolute GPU allowlist and reject forbidden devices."""

    tokens = [token.strip() for token in raw.split(",")]
    if not tokens or any(not token for token in tokens):
        raise ValueError("--gpus must be a comma-separated list such as 1,2")
    gpu_ids: list[int] = []
    for token in tokens:
        if token.startswith("cuda:"):
            token = token.split(":", 1)[1]
        if not token.isdigit():
            raise ValueError(f"invalid GPU identifier {token!r}")
        gpu_id = int(token)
        validate_device(f"cuda:{gpu_id}")
        if gpu_id in gpu_ids:
            raise ValueError(f"duplicate GPU identifier {gpu_id}")
        gpu_ids.append(gpu_id)
    return tuple(gpu_ids)


def _sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def build_wave_jobs(
        spec: Mapping, wave: str, gpu_ids: Sequence[int],
        spec_path: Path = DEFAULT_SPEC,
        lock_path: Path = DEFAULT_LOCK) -> list[Job]:
    """Build a canonical semantic job grid without touching the filesystem."""

    if wave not in WAVES:
        raise ValueError(f"unknown shell-game wave {wave!r}")
    if not gpu_ids:
        raise ValueError("at least one allowed GPU is required")
    devices = tuple(validate_device(f"cuda:{int(gpu_id)}")
                    for gpu_id in gpu_ids)
    stages = tuple(stage.key for stage in CAPACITY_STAGES)
    common = ("--spec", str(spec_path), "--lock", str(lock_path))
    jobs: list[Job] = []

    if wave == "base":
        for index, split in enumerate(SPLITS):
            device = devices[index % len(devices)]
            destination = base_path(spec, split)
            jobs.append(Job(
                name=f"base-{split}",
                command=(
                    PYTHON, "scripts/collect_official_shell_game_base.py",
                    "--split", split, "--device", device, *common,
                ),
                done_file=_sidecar(destination),
                device=device,
            ))
    elif wave == "stages":
        for stage in stages:
            for split in SPLITS:
                destination = stage_path(spec, stage, split)
                jobs.append(Job(
                    name=f"stage-{stage}-{split}",
                    command=(
                        PYTHON,
                        "scripts/prepare_official_shell_game_stage.py",
                        "--stage", stage, "--split", split, *common,
                    ),
                    done_file=_sidecar(destination),
                    device=None,
                ))
    elif wave == "frozen-cache":
        for index, stage in enumerate(stages):
            device = devices[index % len(devices)]
            jobs.append(Job(
                name=f"frozen-cache-{stage}",
                command=(
                    PYTHON, "scripts/cache_official_shell_game_capacity.py",
                    "--stage", stage, "--device", device, *common,
                ),
                done_file=cache_manifest_path(spec, stage),
                device=device,
            ))
    else:
        training = spec["carrier_training"]
        index = 0
        for stage in stages:
            for arm in training["arms"]:
                for seed in training["seeds"]:
                    device = devices[index % len(devices)]
                    jobs.append(Job(
                        name=(f"carrier-{stage}-{arm}-seed-{int(seed)}"),
                        command=(
                            PYTHON,
                            "scripts/train_official_shell_game_capacity.py",
                            "--stage", stage, "--arm", arm,
                            "--seed", str(int(seed)),
                            "--device", device, *common,
                        ),
                        done_file=(carrier_directory(
                            spec, stage, arm, int(seed)) / "manifest.json"),
                        device=device,
                    ))
                    index += 1

    names = [job.name for job in jobs]
    destinations = [job.done_file for job in jobs]
    if len(names) != len(set(names)):
        raise RuntimeError(f"duplicate job name in {wave} wave")
    if len(destinations) != len(set(destinations)):
        raise RuntimeError(f"duplicate output cell in {wave} wave")
    for job in jobs:
        if job.device is not None:
            validate_device(job.device)
    return jobs


def build_plan(
        spec: Mapping, wave: str, gpu_ids: Sequence[int],
        spec_path: Path = DEFAULT_SPEC,
        lock_path: Path = DEFAULT_LOCK) -> list[tuple[str, list[Job]]]:
    """Return dependency-ordered waves; ``all`` never flattens barriers."""

    if wave not in ALL_WAVES:
        raise ValueError(f"unknown shell-game wave {wave!r}")
    selected = WAVES if wave == "all" else (wave,)
    return [
        (name, build_wave_jobs(spec, name, gpu_ids, spec_path, lock_path))
        for name in selected
    ]


def preview_lines(plan: Sequence[tuple[str, Sequence[Job]]]) -> list[str]:
    """Render the pending/read-only plan without creating log directories."""

    lines: list[str] = []
    for wave, jobs in plan:
        for job in jobs:
            status = "complete" if job.done_file.is_file() else "pending"
            lines.append(
                f"{wave}\t{status}\t{job.name}\t{shlex.join(job.command)}")
    return lines


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave", required=True, choices=ALL_WAVES)
    parser.add_argument(
        "--gpus", default="1,2",
        help="absolute GPU ids; only 1 and 2 are permitted (default: 1,2)")
    parser.add_argument("--cpu-jobs", type=int, default=2)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument(
        "--execute", action="store_true",
        help="run the locked plan; without this flag the launcher only previews")
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


def _next_startable(
        queue: Sequence[Job], running: Sequence[tuple],
        cpu_jobs: int) -> int | None:
    busy_devices = {
        process_job.device for _, process_job, _ in running
        if process_job.device is not None
    }
    active_cpu = sum(
        process_job.device is None for _, process_job, _ in running)
    for index, job in enumerate(queue):
        if job.device is None:
            if active_cpu < cpu_jobs:
                return index
        elif job.device not in busy_devices:
            return index
    return None


def _execute_wave(
        spec: Mapping, wave: str, jobs: Sequence[Job],
        cpu_jobs: int) -> None:
    pending = [job for job in jobs if not job.done_file.is_file()]
    if not pending:
        print(f"[shell-game-launch] wave={wave} already complete", flush=True)
        return
    logs = log_root(spec) / wave
    logs.mkdir(parents=True, exist_ok=True)
    environment = _environment()
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
                    job.command, cwd=ROOT, env=environment,
                    stdout=stream, stderr=subprocess.STDOUT)
            except BaseException:
                stream.close()
                raise
            running.append((process, job, stream))
            print(
                f"[shell-game-launch] start wave={wave} job={job.name} "
                f"device={job.device or 'cpu'}", flush=True)
        if not running and queue:
            raise RuntimeError("scheduler cannot start any pending job")
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
                print(f"[shell-game-launch] done {job.name}", flush=True)
            else:
                failed += 1
                print(
                    f"[shell-game-launch] FAIL {job.name} exit={code} "
                    f"log={logs / (job.name + '.log')}", flush=True)
        running = active
    if failed:
        raise SystemExit(
            f"{failed} jobs failed in {wave}; downstream waves were not started")
    print(
        f"[shell-game-launch] wave={wave} complete={completed}", flush=True)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.cpu_jobs < 1:
        raise ValueError("--cpu-jobs must be positive")
    gpu_ids = parse_gpu_ids(args.gpus)
    spec_path = args.spec.resolve()
    lock_path = args.lock.resolve()
    spec = load_locked_spec(spec_path, lock_path)
    plan = build_plan(spec, args.wave, gpu_ids, spec_path, lock_path)
    canonical = sum(len(jobs) for _, jobs in plan)
    pending = sum(
        not job.done_file.is_file() for _, jobs in plan for job in jobs)
    print(
        f"[shell-game-launch] wave={args.wave} canonical={canonical} "
        f"pending={pending} gpus={','.join(map(str, gpu_ids))} "
        f"execute={args.execute}", flush=True)
    if not args.execute:
        for line in preview_lines(plan):
            print(line)
        return
    for wave, jobs in plan:
        _execute_wave(spec, wave, jobs, args.cpu_jobs)


if __name__ == "__main__":
    main()
