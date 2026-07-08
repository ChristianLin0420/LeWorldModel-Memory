#!/usr/bin/env python3
"""Preview or launch the locked five-seed context/rollout extension on GPUs 1/2."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.paper_a_context_rollout_extension_spec import (  # noqa: E402
    CONTEXTS,
    DEFAULT_SPEC,
    EXTENSION_SEEDS,
    OBJECTIVES,
    TASKS,
    extension_directory,
    load_locked_spec,
    repo_path,
    task_record,
    validate_device,
)


PYTHON = sys.executable
WAVES = ("context", "rollout")
ALL_WAVES = (*WAVES, "all")


@dataclass(frozen=True)
class ExtensionJob:
    name: str
    semantic_name: str
    command: tuple[str, ...]
    done_file: Path
    device: str


def parse_gpu_ids(value: str, spec: Mapping) -> tuple[int, ...]:
    tokens = [token.strip() for token in value.split(",")]
    if not tokens or any(not token for token in tokens):
        raise ValueError("--gpus must be a comma-separated list such as 1,2")
    result: list[int] = []
    for token in tokens:
        if token.startswith("cuda:"):
            token = token.split(":", 1)[1]
        if not token.isdigit():
            raise ValueError(f"invalid GPU identifier {token!r}")
        gpu = int(token)
        validate_device(spec, f"cuda:{gpu}")
        if gpu in result:
            raise ValueError(f"duplicate GPU identifier {gpu}")
        result.append(gpu)
    return tuple(result)


def build_wave_jobs(spec: Mapping, wave: str, gpu_ids: Sequence[int],
                    spec_path: Path = DEFAULT_SPEC) -> list[ExtensionJob]:
    if wave not in WAVES:
        raise ValueError(f"unknown extension wave {wave!r}")
    if not gpu_ids:
        raise ValueError("at least one locked GPU is required")
    devices = tuple(validate_device(spec, f"cuda:{int(gpu)}")
                    for gpu in gpu_ids)
    jobs: list[ExtensionJob] = []
    index = 0
    if wave == "context":
        for task in TASKS:
            for history in CONTEXTS:
                variant = f"h{history}"
                for seed in EXTENSION_SEEDS:
                    device = devices[index % len(devices)]
                    directory = extension_directory(
                        spec, "long_context", task, variant, seed)
                    jobs.append(ExtensionJob(
                        name=f"context-{task}-h{history}-seed-{seed}",
                        semantic_name=(f"{task_record(spec, task)['name']} / "
                                       f"context {history} / seed {seed}"),
                        command=(
                            PYTHON,
                            "scripts/run_paper_a_context_rollout_extension.py",
                            "--spec", str(spec_path), "--wave", "long_context",
                            "--task", task, "--context", str(history),
                            "--seed", str(seed), "--device", device, "--execute",
                        ),
                        done_file=directory / "receipt.json",
                        device=device,
                    ))
                    index += 1
    else:
        for task in TASKS:
            for objective in OBJECTIVES:
                for seed in EXTENSION_SEEDS:
                    device = devices[index % len(devices)]
                    directory = extension_directory(
                        spec, "learned_rollout", task, objective, seed)
                    objective_name = ("One-step objective" if objective == "one_step"
                                      else "Eight-step overshooting")
                    jobs.append(ExtensionJob(
                        name=f"rollout-{task}-{objective}-seed-{seed}",
                        semantic_name=(f"{task_record(spec, task)['name']} / "
                                       f"{objective_name} / seed {seed}"),
                        command=(
                            PYTHON,
                            "scripts/run_paper_a_context_rollout_extension.py",
                            "--spec", str(spec_path), "--wave", "learned_rollout",
                            "--task", task, "--objective", objective,
                            "--seed", str(seed), "--device", device, "--execute",
                        ),
                        done_file=directory / "receipt.json",
                        device=device,
                    ))
                    index += 1
    if (len({job.name for job in jobs}) != len(jobs)
            or len({job.done_file for job in jobs}) != len(jobs)):
        raise RuntimeError("extension job grid contains duplicate cells")
    parent_root = repo_path(spec["parent"]["root"], "parent.root")
    output_root = repo_path(spec["output"]["root"], "output.root")
    for job in jobs:
        if output_root not in job.done_file.parents:
            raise RuntimeError(f"job leaves extension root: {job.done_file}")
        if parent_root == job.done_file or parent_root in job.done_file.parents:
            raise RuntimeError(f"job would modify a parent artifact: {job.done_file}")
        validate_device(spec, job.device)
    return jobs


def build_plan(spec: Mapping, wave: str, gpu_ids: Sequence[int],
               spec_path: Path = DEFAULT_SPEC
               ) -> list[tuple[str, list[ExtensionJob]]]:
    if wave not in ALL_WAVES:
        raise ValueError(f"unknown extension plan {wave!r}")
    selected = WAVES if wave == "all" else (wave,)
    return [(name, build_wave_jobs(spec, name, gpu_ids, spec_path))
            for name in selected]


def preview_lines(plan: Sequence[tuple[str, Sequence[ExtensionJob]]]) -> list[str]:
    return [
        f"{wave}\t{'complete' if job.done_file.is_file() else 'pending'}\t"
        f"{job.device}\t{job.semantic_name}\t{shlex.join(job.command)}"
        for wave, jobs in plan for job in jobs
    ]


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.setdefault("PYTHONHASHSEED", "0")
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                     "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment.setdefault(variable, "1")
    return environment


def execute_wave(spec: Mapping, wave: str,
                 jobs: Sequence[ExtensionJob]) -> None:
    queue = [job for job in jobs if not job.done_file.is_file()]
    if not queue:
        print(f"[context-rollout-launch] {wave} already complete", flush=True)
        return
    logs = repo_path(spec["output"]["logs"], "output.logs") / wave
    logs.mkdir(parents=True, exist_ok=True)
    running: list[tuple[subprocess.Popen, ExtensionJob, object]] = []
    failed = 0
    while queue or running:
        busy = {job.device for _, job, _ in running}
        for job in list(queue):
            if job.device in busy:
                continue
            log_path = logs / f"{job.name}.log"
            stream = log_path.open("x")
            process = subprocess.Popen(
                job.command, cwd=ROOT, env=_environment(),
                stdout=stream, stderr=subprocess.STDOUT)
            running.append((process, job, stream))
            queue.remove(job)
            busy.add(job.device)
            print(f"[context-rollout-launch] start {job.semantic_name} "
                  f"on {job.device}", flush=True)
        if not running and queue:
            raise RuntimeError("extension scheduler cannot start pending jobs")
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
                print(f"[context-rollout-launch] FAIL {job.semantic_name} "
                      f"exit={code}", flush=True)
            else:
                print(f"[context-rollout-launch] done {job.semantic_name}",
                      flush=True)
        running = active
    if failed:
        raise SystemExit(
            f"{failed} jobs failed in {wave}; later waves were not started")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--wave", required=True, choices=ALL_WAVES)
    parser.add_argument("--gpus", default="1,2")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    spec = load_locked_spec(args.spec)
    gpu_ids = parse_gpu_ids(args.gpus, spec)
    plan = build_plan(spec, args.wave, gpu_ids, args.spec.resolve())
    count = sum(len(jobs) for _, jobs in plan)
    print(f"[context-rollout-launch] wave={args.wave} jobs={count} "
          f"gpus={gpu_ids} execute={args.execute}", flush=True)
    if not args.execute:
        for line in preview_lines(plan):
            print(line)
        return
    for wave, jobs in plan:
        execute_wave(spec, wave, jobs)


if __name__ == "__main__":
    main()
