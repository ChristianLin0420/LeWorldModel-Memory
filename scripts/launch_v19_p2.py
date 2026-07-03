#!/usr/bin/env python3
"""Launch the V19 P2 development grid (docs/V19_PROPOSAL.md §6, task #P2).

Registered grid: the six development arms (no carrier, the two
action-conditioned recurrent references, both LKC candidates, and the
no-correction control — the V12–V15 bar) on the two development-only task
instances, three seeds each: 6 arms × 2 tasks × 3 seeds = 36 runs, scheduled
across GPUs 0–2. Interventions beyond `lkc_k0` are P3 arms and are *not*
trained here (the dev grid exists to validate telemetry, calibrate gates, and
size P3 via the power analysis — §4.5 Tier 1).

After the grid, runs the probe evaluation + aggregation so the power analysis
lands in outputs/v19_p2/p2_summary.{json,md}.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parents[1]
GPUS = (0, 1, 2)

ARMS = ("none", "acgru", "acssm", "lkc", "lkc_nll", "lkc_k0")
TASKS = ("t1dev", "t3dev")
SEEDS = (0, 1, 2)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/v19_p2")
    parser.add_argument("--host", default="vicreg", choices=("sigreg", "vicreg"),
                        help="P0-selected host (fallback rule, proposal §4.1)")
    parser.add_argument("--jobs-per-gpu", type=int, default=2,
                        help="concurrency per GPU; sized to the measured "
                             "per-job footprint, not optimism")
    parser.add_argument("--project", default="lewm-v19")
    parser.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    parser.add_argument("--no-wandb", dest="wandb", action="store_false")
    return parser.parse_args(argv)


@dataclass
class Job:
    task: str
    arm: str
    seed: int

    @property
    def name(self) -> str:
        return f"{self.task}_{self.arm}_s{self.seed}"


def _job_env(gpu: int, wandb_on: bool) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "EGL_DEVICE_ID": str(gpu),
        "MUJOCO_GL": "egl",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    key_file = REPO / ".wandb.key"
    if wandb_on and key_file.is_file():
        env["WANDB_API_KEY"] = key_file.read_text().strip()
    return env


def _launch(job: Job, gpu: int, args: argparse.Namespace, log_dir: Path
            ) -> subprocess.Popen:
    command = [
        sys.executable, str(REPO / "scripts" / "train_v19_p2.py"),
        "--task", job.task, "--host", args.host, "--arm", job.arm,
        "--seed", str(job.seed), "--output", args.output,
        "--p0-data-root", "outputs/v19_p0_a2/data",
        "--wandb" if args.wandb else "--no-wandb",
        "--wandb-project", args.project,
    ]
    log = open(log_dir / f"{job.name}.log", "w")
    return subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                            env=_job_env(gpu, args.wandb), cwd=REPO)


def _pregenerate_caches(args: argparse.Namespace, log_dir: Path) -> None:
    """Generate each task's data banks once, serially, before any training job.

    The trainers' resolve_banks() lazily creates missing caches; launching a
    9-way first wave without this step made concurrent jobs race on the same
    bank files (digest-mismatch crashes across the first P2 attempt).  One
    renderer per task, sequential, removes the race by construction."""
    for task in TASKS:
        command = [sys.executable, "-c",
                   "import sys; sys.path.insert(0, 'scripts'); "
                   "import train_v19_p2 as t; "
                   f"t.resolve_banks({task!r}, 'outputs/v19_p0_a2/data', "
                   f"{args.output!r} + '/data')"]
        log = open(log_dir / f"data_{task}.log", "w")
        env = _job_env(GPUS[0], wandb_on=False)
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                env=env, cwd=REPO)
        if result.returncode != 0:
            raise SystemExit(f"cache generation failed for {task}; "
                             f"see {log_dir}/data_{task}.log")
        print(f"[v19-p2] cache ready: {task}")


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    log_dir = Path(args.output) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    _pregenerate_caches(args, log_dir)

    queue = [Job(t, a, s) for t, a, s in product(TASKS, ARMS, SEEDS)]
    queue = [job for job in queue
             if not (Path(args.output) / job.task / job.arm / f"s{job.seed}"
                     / "gates.json").is_file()]
    print(f"[v19-p2] {len(queue)} jobs to run "
          f"({args.jobs_per_gpu}/GPU over GPUs {GPUS})")

    running: list[tuple[subprocess.Popen, Job, int]] = []
    gpu_load = {gpu: 0 for gpu in GPUS}
    done = crashed = 0
    while queue or running:
        while queue and min(gpu_load.values()) < args.jobs_per_gpu:
            gpu = min(GPUS, key=gpu_load.__getitem__)
            job = queue.pop(0)
            running.append((_launch(job, gpu, args, log_dir), job, gpu))
            gpu_load[gpu] += 1
        time.sleep(20)
        still = []
        for proc, job, gpu in running:
            code = proc.poll()
            if code is None:
                still.append((proc, job, gpu))
                continue
            gpu_load[gpu] -= 1
            done += code == 0
            crashed += code != 0
            if code != 0:
                print(f"[v19-p2] CRASH {job.name} (exit {code})")
                # let the dying process release GPU memory before the slot
                # is refilled (the first P2 attempt OOMed on this transient)
                time.sleep(30)
        running = still
        tags = ",".join(f"{j.name}@gpu{g}" for _, j, g in running)
        print(f"\r[v19-p2] {done} done, {crashed} crashed | running: {tags}",
              end="", flush=True)
    print(f"\n[v19-p2] grid finished: {done} ok, {crashed} crashed")

    evaluate = [sys.executable, str(REPO / "scripts" / "eval_v19_p2.py"),
                "--root", args.output]
    aggregate = [sys.executable, str(REPO / "scripts" / "aggregate_v19_p2.py"),
                 "--root", args.output]
    subprocess.run(evaluate, cwd=REPO, check=False)
    subprocess.run(aggregate, cwd=REPO, check=False)


if __name__ == "__main__":
    main()
