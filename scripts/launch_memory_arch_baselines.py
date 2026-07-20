#!/usr/bin/env python3
"""Launch GRU/LSTM/Mamba-lite memory baselines across OGBench cells."""

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
RUNNER = ROOT / "scripts" / "run_multiview_patchset_memory_baseline.py"
DEFAULT_OUTPUT = ROOT / "outputs" / "memory_arch_baselines_v1"
DEFAULT_CACHE_ROOT = ROOT / "outputs" / "multiview_patchset_color_jepa_native_v1"
DEFAULT_BASELINES = ("gru", "lstm", "mamba_lite")
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
DEFAULT_AGES = (4, 8, 15)
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_GPUS = ("0", "1", "2")


def python_bin() -> str:
    candidate = ROOT / ".venv" / "bin" / "python"
    return str(candidate if candidate.exists() else Path(sys.executable))


def env_key(env_name: str) -> str:
    return env_name.replace("/", "_")


@dataclass(frozen=True)
class Job:
    baseline: str
    env_name: str
    age: int
    seed: int

    @property
    def label(self) -> str:
        return f"{self.baseline}_{env_key(self.env_name)}_age{self.age}_s{self.seed}"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--baselines", nargs="*", default=list(DEFAULT_BASELINES))
    parser.add_argument("--envs", nargs="*", default=list(DEFAULT_ENVS))
    parser.add_argument("--ages", type=int, nargs="*", default=list(DEFAULT_AGES))
    parser.add_argument("--seeds", type=int, nargs="*", default=list(DEFAULT_SEEDS))
    parser.add_argument("--gpus", nargs="*", default=list(DEFAULT_GPUS))
    parser.add_argument("--episodes", type=int, default=384)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--dim", type=int, default=160)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--cache-k", type=int, default=8)
    parser.add_argument("--cue-mode", default="color")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


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


def baseline_root(args: argparse.Namespace, baseline: str) -> Path:
    return args.output / baseline


def result_path(args: argparse.Namespace, job: Job) -> Path:
    return baseline_root(args, job.baseline) / env_key(job.env_name) / f"age_{job.age}" / f"s{job.seed}" / "result.json"


def runner_base(args: argparse.Namespace, job: Job | None, baseline: str, env_name: str) -> list[str]:
    return [
        python_bin(),
        str(RUNNER),
        "--output",
        str(baseline_root(args, baseline)),
        "--cache-root",
        str(args.cache_root),
        "--baseline",
        baseline,
        "--env-name",
        env_name,
        "--episodes",
        str(args.episodes),
        "--img-size",
        str(args.img_size),
        "--dim",
        str(args.dim),
        "--slots",
        str(args.slots),
        "--heads",
        str(args.heads),
        "--hidden",
        str(args.hidden),
        "--chunk",
        str(args.chunk),
        "--cache-k",
        str(args.cache_k),
        "--cue-mode",
        str(args.cue_mode),
    ]


def command_for(args: argparse.Namespace, job: Job) -> list[str]:
    return [
        *runner_base(args, job, job.baseline, job.env_name),
        "--age",
        str(job.age),
        "--seed",
        str(job.seed),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--device",
        "cuda:0",
    ]


def aggregate(args: argparse.Namespace, baseline: str) -> None:
    command = [
        *runner_base(args, None, baseline, args.envs[0] if args.envs else DEFAULT_ENVS[0]),
        "--aggregate",
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def write_monitor(args: argparse.Namespace, *, total: int, running: dict[str, tuple[Job, subprocess.Popen[bytes], object]], failed: list[dict[str, object]]) -> None:
    completed = 0
    per_baseline: dict[str, int] = {}
    for baseline in args.baselines:
        count = sum(1 for _ in baseline_root(args, baseline).glob("*/*/s*/result.json"))
        per_baseline[baseline] = int(count)
        completed += int(count)
    monitor = {
        "schema": "memory_arch_baseline_monitor_v1",
        "status": "failed" if failed else ("running" if running or completed < total else "completed"),
        "updated_unix": time.time(),
        "total_jobs": int(total),
        "completed_jobs": int(completed),
        "running": [
            {
                "gpu": gpu,
                "baseline": job.baseline,
                "env_name": job.env_name,
                "age": int(job.age),
                "seed": int(job.seed),
                "pid": int(process.pid),
            }
            for gpu, (job, process, _) in sorted(running.items())
        ],
        "per_baseline_completed": per_baseline,
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
    queue = [
        Job(str(baseline), str(env_name), int(age), int(seed))
        for baseline in args.baselines
        for env_name in args.envs
        for age in args.ages
        for seed in args.seeds
    ]
    if args.resume:
        queue = [job for job in queue if not result_path(args, job).is_file()]
    total = len(args.baselines) * len(args.envs) * len(args.ages) * len(args.seeds)
    receipt = {
        "schema": "memory_arch_baseline_launch_v1",
        "output": str(args.output),
        "cache_root": str(args.cache_root),
        "baselines": list(args.baselines),
        "envs": list(args.envs),
        "ages": [int(v) for v in args.ages],
        "seeds": [int(v) for v in args.seeds],
        "gpus": list(args.gpus),
        "total_jobs": int(total),
        "queued_jobs": int(len(queue)),
        "episodes": int(args.episodes),
        "epochs": int(args.epochs),
        "launched_unix": time.time(),
    }
    (args.output / "launch_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if args.dry_run:
        for job in queue:
            print("[baseline-launch] job", job)
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
                f"[baseline-launch] started gpu={gpu} pid={process.pid} "
                f"baseline={job.baseline} env={job.env_name} age={job.age} seed={job.seed}",
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
                f"[baseline-launch] finished gpu={gpu} code={code} "
                f"baseline={job.baseline} env={job.env_name} age={job.age} seed={job.seed}",
                flush=True,
            )
            if code != 0:
                failed.append(
                    {
                        "gpu": gpu,
                        "baseline": job.baseline,
                        "env_name": job.env_name,
                        "age": int(job.age),
                        "seed": int(job.seed),
                        "code": int(code),
                    }
                )
        write_monitor(args, total=total, running=running, failed=failed)
        if failed:
            for _, process, stream in running.values():
                if process.poll() is None:
                    process.terminate()
                stream.close()
            raise SystemExit(f"failed jobs: {failed}")
    for baseline in args.baselines:
        aggregate(args, str(baseline))
    write_monitor(args, total=total, running=running, failed=failed)


if __name__ == "__main__":
    main()
