#!/usr/bin/env python3
"""Launch LeWM PushT host-writer cells across GPUs with cache-safe ordering."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_lewm_pusht_host_writer.py"
DEFAULT_OUTPUT = ROOT / "outputs/lewm_pusht_host_writer_counterfactual_agegrid_v1"


def python_bin() -> str:
    candidate = ROOT / ".venv/bin/python"
    return str(candidate if candidate.exists() else Path(sys.executable))


def result_path(output: Path, task: str, seed: int, age: int) -> Path:
    return output / task / f"s{int(seed)}" / f"age_{int(age)}" / "result.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--gpus", nargs="*", default=["0", "1", "2"])
    parser.add_argument("--tasks", nargs="*", default=[
        "transient-visual-token-recall",
        "multi-item-visual-binding-recall",
    ])
    parser.add_argument("--ages", type=int, nargs="*", default=[4, 8, 15])
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    parser.add_argument("--cache-seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--slots", type=int, default=6)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--candidate-count", type=int, default=6)
    parser.add_argument("--target-mode", default="counterfactual_delta_flat",
                        choices=["counterfactual_delta_flat"])
    parser.add_argument("--counterfactual-cache",
                        default=str(ROOT / "outputs/lewm_pusht_counterfactual_cue_cache_v1"))
    parser.add_argument("--frame-batch-size", type=int, default=128)
    parser.add_argument("--host-weight", type=float, default=1.0)
    parser.add_argument("--context-weight", type=float, default=1.0)
    parser.add_argument("--memory-weight", type=float, default=1.0)
    parser.add_argument("--residual-l2-weight", type=float, default=1.0e-4)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def command_for(args: argparse.Namespace, output: Path, task: str,
                age: int, seed: int) -> list[str]:
    return [
        python_bin(), str(RUNNER),
        "--output", str(output),
        "--task", task,
        "--age", str(age),
        "--seed", str(seed),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--eval-batch-size", str(args.eval_batch_size),
        "--dim", str(args.dim),
        "--slots", str(args.slots),
        "--heads", str(args.heads),
        "--candidate-count", str(args.candidate_count),
        "--target-mode", args.target_mode,
        "--counterfactual-cache", str(args.counterfactual_cache),
        "--frame-batch-size", str(args.frame_batch_size),
        "--host-weight", str(args.host_weight),
        "--context-weight", str(args.context_weight),
        "--memory-weight", str(args.memory_weight),
        "--residual-l2-weight", str(args.residual_l2_weight),
        "--device", "cuda:0",
    ]


def run_queue(args: argparse.Namespace, output: Path,
              jobs: list[tuple[str, int, int]], phase: str) -> None:
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    pending = [
        job for job in jobs
        if not result_path(output, job[0], job[2], job[1]).is_file()
    ]
    print(f"[lewm-launch] phase={phase} pending={len(pending)}", flush=True)
    if args.dry_run:
        for task, age, seed in pending:
            print(f"{phase}: task={task} age={age} seed={seed}")
        return
    running: dict[str, tuple[subprocess.Popen[bytes], tuple[str, int, int], object]] = {}
    while pending or running:
        for gpu in args.gpus:
            if gpu in running or not pending:
                continue
            task, age, seed = pending.pop(0)
            log_path = logs / f"{phase}_{task}_age{age}_s{seed}_gpu{gpu}.log"
            stream = log_path.open("wb")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            process = subprocess.Popen(
                command_for(args, output, task, age, seed),
                cwd=str(ROOT), env=env, stdout=stream, stderr=subprocess.STDOUT)
            running[gpu] = (process, (task, age, seed), stream)
            print(
                f"[lewm-launch] started phase={phase} gpu={gpu} "
                f"task={task} age={age} seed={seed} pid={process.pid} "
                f"log={log_path}",
                flush=True,
            )
        time.sleep(float(args.poll_seconds))
        for gpu, (process, job, stream) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            stream.close()
            task, age, seed = job
            del running[gpu]
            if code != 0:
                for other, (_, _, other_stream) in list(running.items()):
                    other_stream.close()
                    print(f"[lewm-launch] still-running gpu={other}", flush=True)
                raise SystemExit(
                    f"failed phase={phase} gpu={gpu} task={task} "
                    f"age={age} seed={seed} code={code}")
            print(
                f"[lewm-launch] completed phase={phase} gpu={gpu} "
                f"task={task} age={age} seed={seed}",
                flush=True,
            )


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cache_seed_jobs = [
        (task, age, int(args.cache_seed))
        for task in args.tasks
        for age in args.ages
        if int(args.cache_seed) in args.seeds
    ]
    remaining_jobs = [
        (task, age, seed)
        for seed in args.seeds
        if int(seed) != int(args.cache_seed)
        for task in args.tasks
        for age in args.ages
    ]
    run_queue(args, output, cache_seed_jobs, "cache_seed")
    run_queue(args, output, remaining_jobs, "main")
    if not args.dry_run:
        subprocess.check_call([
            python_bin(), str(RUNNER),
            "--aggregate",
            "--output", str(output),
            "--target-mode", args.target_mode,
            "--tasks", *args.tasks,
            "--ages", *[str(value) for value in args.ages],
            "--seeds", *[str(value) for value in args.seeds],
        ], cwd=str(ROOT))


if __name__ == "__main__":
    main()
