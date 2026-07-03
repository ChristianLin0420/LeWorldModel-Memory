#!/usr/bin/env python3
"""Schedule the V19 P0 host-preflight grid: 2 hosts x 4 tasks x 3 seeds.

First builds any missing per-task data caches (scripts/make_v19_p0_data.py,
parallel across GPUs for EGL rendering), then runs the 24 training cells over
GPUs 0-2 with THREE concurrent jobs per GPU (9-way parallel), pinning
CUDA_VISIBLE_DEVICES / MUJOCO_GL=egl / EGL_DEVICE_ID per job.  Job stdout goes
to per-job logfiles under <output>/logs/ with a one-line live status; when the
grid finishes, scripts/aggregate_v19_p0.py builds the summary and applies the
per-task attribution rule.
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

from scripts.make_v19_p0_data import DEFAULT_ROOT as DEFAULT_DATA_ROOT
from scripts.make_v19_p0_data import P0_TASKS, _cache_valid, episode_sizes, task_bank_paths
from scripts.train_v19_p0 import HOSTS

SEEDS = (0, 1, 2)
GPUS = (0, 1, 2)
MAX_JOBS_PER_GPU = 3
POLL_SECONDS = 10.0


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/v19_p0")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--project", default="lewm-v19")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    parser.add_argument("--no-wandb", dest="wandb", action="store_false")
    return parser.parse_args(argv)


def _job_env(gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "MUJOCO_GL": "egl",
                "EGL_DEVICE_ID": str(gpu)})
    key_path = ROOT / ".wandb.key"
    if "WANDB_API_KEY" not in env and key_path.exists():
        env["WANDB_API_KEY"] = key_path.read_text().strip()  # never printed
    return env


def _cache_complete(data_root: str, task: str) -> bool:
    train_episodes, val_episodes = episode_sizes()
    paths = task_bank_paths(data_root, task, train_episodes, val_episodes)
    return all(_cache_valid(paths[split][view])
               for split in ("train", "val") for view in ("clean", "observed"))


def _ensure_caches(args: argparse.Namespace, log_dir: Path) -> None:
    """Generate missing task caches, one subprocess per task, GPU-pinned EGL."""
    missing = [task for task in P0_TASKS
               if not _cache_complete(args.data_root, task)]
    if not missing:
        print("[v19-p0] all data caches present and hash-verified", flush=True)
        return
    print(f"[v19-p0] generating data caches for: {', '.join(missing)}", flush=True)
    processes: list[tuple[str, subprocess.Popen, object]] = []
    for index, task in enumerate(missing):
        gpu = GPUS[index % len(GPUS)]
        handle = (log_dir / f"data_{task}.log").open("w")
        process = subprocess.Popen(
            [sys.executable, str(ROOT / "scripts" / "make_v19_p0_data.py"),
             "--task", task, "--root", args.data_root],
            stdout=handle, stderr=subprocess.STDOUT, cwd=ROOT, env=_job_env(gpu))
        processes.append((task, process, handle))
    failures = []
    for task, process, handle in processes:
        code = process.wait()
        handle.close()
        if code != 0:
            failures.append(task)
    if failures:
        raise RuntimeError(
            f"data cache generation failed for {failures} (see {log_dir})")


@dataclass
class Job:
    task: str
    host: str
    seed: int
    gpu: int | None = None
    process: subprocess.Popen | None = None
    log_handle: object = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return f"{self.task}_{self.host}_s{self.seed}"


def _launch(job: Job, gpu: int, args: argparse.Namespace, log_dir: Path) -> None:
    command = [sys.executable, str(ROOT / "scripts" / "train_v19_p0.py"),
               "--task", job.task, "--host", job.host, "--seed", str(job.seed),
               "--output", args.output, "--data-root", args.data_root,
               "--wandb-project", args.project,
               "--wandb" if args.wandb else "--no-wandb"]
    if args.entity:
        command += ["--wandb-entity", args.entity]
    job.gpu = gpu
    job.log_handle = (log_dir / f"{job.name}.log").open("w")
    job.process = subprocess.Popen(command, stdout=job.log_handle,
                                   stderr=subprocess.STDOUT, cwd=ROOT,
                                   env=_job_env(gpu))


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    log_dir = Path(args.output) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    _ensure_caches(args, log_dir)

    # Seed-major order interleaves hosts and tasks, so each GPU carries a mix
    # of heavy (sigreg, D=192/12L) and light (vicreg, D=128/6L) cells.
    pending = [Job(task, host, seed)
               for seed in SEEDS for task in P0_TASKS for host in HOSTS]
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
        print(f"\r[v19-p0] {done}/{total} done, {len(failed)} crashed | "
              f"running: {active:<120}", end="", flush=True)
        if running:
            time.sleep(POLL_SECONDS)
    print()
    if failed:
        print(f"[v19-p0] crashed jobs (see {log_dir}): {', '.join(failed)}")

    subprocess.run([sys.executable, str(ROOT / "scripts" / "aggregate_v19_p0.py"),
                    "--root", args.output], cwd=ROOT, check=False)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
