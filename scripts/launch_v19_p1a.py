#!/usr/bin/env python3
"""Schedule the full V19 P1a certificate grid: 6 tasks x 3 seeds over 3 GPUs.

Each job runs scripts/run_v19_p1a.py in its own process with
CUDA_VISIBLE_DEVICES / EGL_DEVICE_ID pinned to one GPU; at most two jobs share
a GPU (EGL rendering is light), so six jobs run in parallel.  Job stdout goes
to per-job logfiles under <output>/logs/ and a one-line live status is printed.
When the grid finishes, scripts/aggregate_v19_p1a.py builds the summary table.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.tasks_v19 import TASKS

SEEDS = (0, 1, 2)
GPUS = (0, 1, 2)
MAX_JOBS_PER_GPU = 2
POLL_SECONDS = 3.0


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/v19_p1a")
    parser.add_argument("--project", default="lewm-v19")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    parser.add_argument("--no-wandb", dest="wandb", action="store_false")
    return parser.parse_args(argv)


@dataclass
class Job:
    task: str
    seed: int
    gpu: int | None = None
    process: subprocess.Popen | None = None
    log_handle: object = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return f"{self.task}_s{self.seed}"


def _launch(job: Job, gpu: int, args: argparse.Namespace, log_dir: Path) -> None:
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "MUJOCO_GL": "egl",
                "EGL_DEVICE_ID": str(gpu)})
    command = [sys.executable, str(ROOT / "scripts" / "run_v19_p1a.py"),
               "--task", job.task, "--seed", str(job.seed),
               "--output", args.output, "--project", args.project,
               "--wandb" if args.wandb else "--no-wandb"]
    if args.entity:
        command += ["--entity", args.entity]
    job.gpu = gpu
    job.log_handle = (log_dir / f"{job.name}.log").open("w")
    job.process = subprocess.Popen(command, stdout=job.log_handle,
                                   stderr=subprocess.STDOUT, cwd=ROOT, env=env)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    log_dir = Path(args.output) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = [Job(task, seed) for task in TASKS for seed in SEEDS]
    running: list[Job] = []
    failed: list[str] = []
    done = 0
    total = len(pending)
    gpu_load = {gpu: 0 for gpu in GPUS}

    while pending or running:
        for job in list(running):
            if job.process.poll() is None:
                continue
            running.remove(job)
            gpu_load[job.gpu] -= 1
            job.log_handle.close()
            done += 1
            if job.process.returncode != 0:
                failed.append(job.name)
        while pending:
            gpu = min(GPUS, key=gpu_load.__getitem__)
            if gpu_load[gpu] >= MAX_JOBS_PER_GPU:
                break
            job = pending.pop(0)
            _launch(job, gpu, args, log_dir)
            gpu_load[gpu] += 1
            running.append(job)
        active = ",".join(f"{job.name}@gpu{job.gpu}" for job in running)
        print(f"\r[v19-p1a] {done}/{total} done, {len(failed)} crashed | "
              f"running: {active:<80}", end="", flush=True)
        if running:
            time.sleep(POLL_SECONDS)
    print()
    if failed:
        print(f"[v19-p1a] crashed jobs (see {log_dir}): {', '.join(failed)}")

    subprocess.run([sys.executable, str(ROOT / "scripts" / "aggregate_v19_p1a.py"),
                    "--root", args.output], cwd=ROOT, check=False)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
