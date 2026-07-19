#!/usr/bin/env python3
"""Launch OGBench feature-host memory stages on GPUs 0, 1, and 2."""

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
RUNNER = ROOT / "scripts" / "run_ogbench_feature_host_stage.py"
DEFAULT_OUTPUT = ROOT / "outputs" / "ogbench_feature_host_stage_v1"
DEFAULT_ENVS = ("antmaze-large-navigate-v0", "cube-single-play-v0")
DEFAULT_AGES = (4, 8, 15)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_GPUS = ("0", "1", "2")


def python_bin() -> str:
    candidate = ROOT / ".venv" / "bin" / "python"
    return str(candidate if candidate.exists() else Path(sys.executable))


@dataclass(frozen=True)
class Job:
    env_name: str
    age: int
    seed: int

    @property
    def key(self) -> str:
        return self.env_name.replace("/", "_")

    @property
    def label(self) -> str:
        return f"{self.key}_age{self.age}_s{self.seed}"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--envs", nargs="*", default=list(DEFAULT_ENVS))
    parser.add_argument("--ages", type=int, nargs="*", default=list(DEFAULT_AGES))
    parser.add_argument("--seeds", type=int, nargs="*", default=list(DEFAULT_SEEDS))
    parser.add_argument("--gpus", nargs="*", default=list(DEFAULT_GPUS))
    parser.add_argument("--train-bases", type=int, default=200)
    parser.add_argument("--validation-bases", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def env_for_gpu(gpu: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "EGL_DEVICE_ID": str(gpu),
        "MUJOCO_GL": "egl",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    return env


def result_path(output: Path, job: Job) -> Path:
    return output / job.key / f"age_{job.age}" / f"s{job.seed}" / "result.json"


def runner_base(args: argparse.Namespace, env_name: str) -> list[str]:
    return [
        python_bin(), str(RUNNER),
        "--output", str(args.output),
        "--env-name", env_name,
        "--train-bases", str(args.train_bases),
        "--validation-bases", str(args.validation_bases),
        "--feature-batch-size", str(args.feature_batch_size),
        "--dim", str(args.dim),
    ]


def prepare_caches(args: argparse.Namespace) -> None:
    log_dir = args.output / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for index, env_name in enumerate(args.envs):
        gpu = args.gpus[index % len(args.gpus)]
        log_path = log_dir / f"prepare_{env_name}.log"
        command = [
            *runner_base(args, env_name),
            "--prepare-cache",
            "--device", "cuda:0",
        ]
        if args.dry_run:
            print("[ogbench-host-launch] prepare", gpu, " ".join(command))
            continue
        with log_path.open("wb") as log:
            print(f"[ogbench-host-launch] preparing env={env_name} gpu={gpu}", flush=True)
            subprocess.run(command, cwd=ROOT, env=env_for_gpu(gpu),
                           stdout=log, stderr=subprocess.STDOUT, check=True)


def command_for(args: argparse.Namespace, job: Job) -> list[str]:
    return [
        *runner_base(args, job.env_name),
        "--age", str(job.age),
        "--seed", str(job.seed),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--device", "cuda:0",
    ]


def aggregate(args: argparse.Namespace) -> None:
    env_name = args.envs[0] if args.envs else DEFAULT_ENVS[0]
    command = [
        python_bin(), str(RUNNER),
        "--output", str(args.output),
        "--env-name", env_name,
        "--aggregate",
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    log_dir = args.output / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    prepare_caches(args)
    queue = [
        Job(env_name=env_name, age=int(age), seed=int(seed))
        for env_name in args.envs
        for age in args.ages
        for seed in args.seeds
        if not (args.resume and result_path(args.output, Job(env_name, int(age), int(seed))).is_file())
    ]
    receipt = {
        "schema": "ogbench_feature_host_stage_launch_v1",
        "output": str(args.output),
        "envs": list(args.envs),
        "ages": [int(v) for v in args.ages],
        "seeds": [int(v) for v in args.seeds],
        "gpus": list(args.gpus),
        "pending_jobs": len(queue),
        "launched_unix": time.time(),
    }
    (args.output / "launch_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if args.dry_run:
        for job in queue:
            print("[ogbench-host-launch] job", job)
        return

    running: dict[str, tuple[Job, subprocess.Popen[bytes], object]] = {}
    failed: list[dict[str, object]] = []
    while queue or running:
        for gpu in args.gpus:
            if gpu in running or not queue:
                continue
            job = queue.pop(0)
            log_path = log_dir / f"{job.label}_gpu{gpu}.log"
            stream = log_path.open("wb")
            process = subprocess.Popen(
                command_for(args, job), cwd=ROOT, env=env_for_gpu(gpu),
                stdout=stream, stderr=subprocess.STDOUT)
            running[gpu] = (job, process, stream)
            print(
                f"[ogbench-host-launch] started gpu={gpu} pid={process.pid} "
                f"env={job.env_name} age={job.age} seed={job.seed}",
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
                f"[ogbench-host-launch] finished gpu={gpu} code={code} "
                f"env={job.env_name} age={job.age} seed={job.seed}",
                flush=True,
            )
            if code != 0:
                failed.append({
                    "gpu": gpu,
                    "env_name": job.env_name,
                    "age": job.age,
                    "seed": job.seed,
                    "code": int(code),
                })
        if failed:
            for _, process, stream in running.values():
                if process.poll() is None:
                    process.terminate()
                stream.close()
            (args.output / "failure.json").write_text(
                json.dumps({"failed": failed}, indent=2, sort_keys=True) + "\n")
            raise SystemExit(f"failed jobs: {failed}")
    aggregate(args)


if __name__ == "__main__":
    main()
