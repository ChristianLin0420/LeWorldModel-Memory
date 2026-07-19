#!/usr/bin/env python3
"""Launch mixed-age slot-memory JEPA cells across OGBench environments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_multiview_patchset_auto_age.py"
DEFAULT_OUTPUT = ROOT / "outputs" / "multiview_patchset_auto_age_v1"
DEFAULT_CACHE_ROOT = ROOT / "outputs" / "multiview_patchset_color_jepa_native_v1"
DEFAULT_ENVS = (
    "pointmaze-medium-navigate-v0",
    "pointmaze-large-navigate-v0",
    "pointmaze-giant-navigate-v0",
    "pointmaze-teleport-navigate-v0",
    "antmaze-large-navigate-v0",
    "antmaze-giant-navigate-v0",
    "humanoidmaze-large-navigate-v0",
    "cube-single-play-v0",
    "cube-double-play-v0",
    "cube-triple-play-v0",
    "puzzle-3x3-play-v0",
    "scene-play-v0",
)
DEFAULT_TRAIN_AGES = (4, 8, 15)
DEFAULT_EVAL_AGES = (4, 6, 8, 10, 12, 15, 18)
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_GPUS = ("0", "1", "2")


@dataclass(frozen=True)
class Job:
    env_name: str
    seed: int

    @property
    def label(self) -> str:
        return f"{self.env_name.replace('/', '_')}_s{self.seed}"


def python_bin() -> str:
    candidate = ROOT / ".venv" / "bin" / "python"
    return str(candidate if candidate.exists() else Path(sys.executable))


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--envs", nargs="*", default=list(DEFAULT_ENVS))
    parser.add_argument("--train-ages", type=int, nargs="*", default=list(DEFAULT_TRAIN_AGES))
    parser.add_argument("--eval-ages", type=int, nargs="*", default=list(DEFAULT_EVAL_AGES))
    parser.add_argument("--seeds", type=int, nargs="*", default=list(DEFAULT_SEEDS))
    parser.add_argument("--gpus", nargs="*", default=list(DEFAULT_GPUS))
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--dim", type=int, default=160)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def result_path(args: argparse.Namespace, job: Job) -> Path:
    return args.output / job.env_name.replace("/", "_") / "auto_age" / f"s{int(job.seed)}" / "result.json"


def command_for(args: argparse.Namespace, job: Job) -> list[str]:
    return [
        python_bin(),
        str(RUNNER),
        "--output",
        str(args.output),
        "--cache-root",
        str(args.cache_root),
        "--env-name",
        job.env_name,
        "--seed",
        str(job.seed),
        "--train-ages",
        *[str(value) for value in args.train_ages],
        "--eval-ages",
        *[str(value) for value in args.eval_ages],
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--dim",
        str(args.dim),
        "--slots",
        str(args.slots),
        "--heads",
        str(args.heads),
        "--device",
        "cuda:0",
    ]


def env_for_gpu(gpu: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "EGL_DEVICE_ID": str(gpu),
            "MUJOCO_GL": "egl",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    return env


def write_monitor(args: argparse.Namespace, *, total: int, running: dict[str, tuple[Job, subprocess.Popen[bytes], object]], failed: list[dict[str, object]]) -> None:
    completed = sum(1 for _ in args.output.glob("*/*/s*/result.json"))
    monitor = {
        "schema": "multiview_patchset_auto_age_monitor_v1",
        "status": "failed" if failed else ("running" if running or completed < total else "completed"),
        "updated_unix": time.time(),
        "total_jobs": int(total),
        "completed_jobs": int(completed),
        "running": [
            {
                "gpu": gpu,
                "env_name": job.env_name,
                "seed": int(job.seed),
                "pid": int(process.pid),
            }
            for gpu, (job, process, _) in sorted(running.items())
        ],
        "failed": failed,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "monitor_status.json").write_text(json.dumps(monitor, indent=2, sort_keys=True) + "\n")


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    args.output = args.output if args.output.is_absolute() else ROOT / args.output
    args.cache_root = args.cache_root if args.cache_root.is_absolute() else ROOT / args.cache_root
    args.output.mkdir(parents=True, exist_ok=True)
    log_dir = args.output / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    queue = [Job(str(env_name), int(seed)) for env_name in args.envs for seed in args.seeds]
    if args.resume:
        queue = [job for job in queue if not result_path(args, job).is_file()]
    total = len(args.envs) * len(args.seeds)
    receipt = {
        "schema": "multiview_patchset_auto_age_launch_v1",
        "output": str(args.output),
        "cache_root": str(args.cache_root),
        "envs": list(args.envs),
        "train_ages": [int(value) for value in args.train_ages],
        "eval_ages": [int(value) for value in args.eval_ages],
        "seeds": [int(value) for value in args.seeds],
        "gpus": list(args.gpus),
        "total_jobs": int(total),
        "queued_jobs": int(len(queue)),
        "epochs": int(args.epochs),
        "launched_unix": time.time(),
    }
    (args.output / "launch_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if args.dry_run:
        for job in queue:
            print("[auto-age-launch] job", job)
        return
    running: dict[str, tuple[Job, subprocess.Popen[bytes], object]] = {}
    failed: list[dict[str, object]] = []
    write_monitor(args, total=total, running=running, failed=failed)
    while queue or running:
        for gpu in args.gpus:
            if gpu in running or not queue:
                continue
            job = queue.pop(0)
            log_path = log_dir / f"{job.label}_gpu{gpu}.log"
            stream = log_path.open("wb")
            process = subprocess.Popen(
                command_for(args, job),
                cwd=ROOT,
                env=env_for_gpu(gpu),
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            running[gpu] = (job, process, stream)
            print(
                f"[auto-age-launch] started gpu={gpu} pid={process.pid} "
                f"env={job.env_name} seed={job.seed}",
                flush=True,
            )
        time.sleep(float(args.poll_seconds))
        for gpu, (job, process, stream) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            stream.close()
            del running[gpu]
            print(
                f"[auto-age-launch] finished gpu={gpu} code={code} "
                f"env={job.env_name} seed={job.seed}",
                flush=True,
            )
            if code != 0:
                failed.append({"gpu": gpu, "env_name": job.env_name, "seed": int(job.seed), "code": int(code)})
        write_monitor(args, total=total, running=running, failed=failed)
        if failed:
            break
    if not failed:
        write_monitor(args, total=total, running=running, failed=failed)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
