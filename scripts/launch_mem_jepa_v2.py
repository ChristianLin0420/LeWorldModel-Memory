#!/usr/bin/env python3
"""Launch Mem-JEPA v2 cue-card diagnostics on GPUs 0, 1, and 2."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
RUNNER = ROOT / "scripts" / "run_mem_jepa_v2.py"
GPUS = (0, 1, 2)
AGES = (4, 8, 15)


@dataclass(frozen=True)
class Job:
    age: int
    seed: int
    gpu: int

    @property
    def name(self) -> str:
        return f"age{self.age}_seed{self.seed}"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "mem_jepa_v2")
    parser.add_argument("--episodes", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--detach", action="store_true")
    return parser.parse_args(argv)


def env_for_gpu(gpu: int) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "EGL_DEVICE_ID": str(gpu),
        "MUJOCO_GL": "egl",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    return env


def run_prepare(args: argparse.Namespace) -> None:
    log_dir = args.output / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(PYTHON), str(RUNNER),
        "--output", str(args.output),
        "--episodes", str(args.episodes),
        "--seed", str(args.seed),
        "--prepare-data",
    ]
    with open(log_dir / "prepare_data.log", "w") as log:
        subprocess.run(command, cwd=ROOT, env=env_for_gpu(GPUS[1]),
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def launch(job: Job, args: argparse.Namespace) -> subprocess.Popen:
    run_dir = args.output / "runs" / job.name
    if args.resume and (run_dir / "summary.json").exists():
        return subprocess.Popen([sys.executable, "-c", "pass"])
    log_dir = args.output / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(PYTHON), str(RUNNER),
        "--output", str(args.output),
        "--age", str(job.age),
        "--seed", str(job.seed),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
    ]
    log = open(log_dir / f"{job.name}.log", "w")
    return subprocess.Popen(command, cwd=ROOT, env=env_for_gpu(job.gpu),
                            stdout=log, stderr=subprocess.STDOUT)


def aggregate(args: argparse.Namespace) -> None:
    subprocess.run([str(PYTHON), str(RUNNER), "--output", str(args.output), "--aggregate"],
                   cwd=ROOT, check=True)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    run_prepare(args)
    jobs = [Job(age=age, seed=args.seed, gpu=gpu)
            for age, gpu in zip(AGES, GPUS, strict=True)]
    processes = [(job, launch(job, args)) for job in jobs]
    receipt = {
        "schema": "mem_jepa_v2_2_launch_v1",
        "output": str(args.output),
        "jobs": [
            {"age": job.age, "seed": job.seed, "gpu": job.gpu,
             "pid": proc.pid, "name": job.name}
            for job, proc in processes
        ],
    }
    (args.output / "launch_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)
    if args.detach:
        return
    failed = 0
    while processes:
        remaining = []
        for job, proc in processes:
            code = proc.poll()
            if code is None:
                remaining.append((job, proc))
                continue
            failed += int(code != 0)
            print(f"[mem-jepa-v2] {job.name} gpu{job.gpu} exit={code}", flush=True)
        processes = remaining
        if processes:
            time.sleep(10)
    if failed:
        raise SystemExit(f"{failed} Mem-JEPA v2 jobs failed")
    aggregate(args)


if __name__ == "__main__":
    main()
