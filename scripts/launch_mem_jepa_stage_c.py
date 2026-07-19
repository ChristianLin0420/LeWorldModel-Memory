#!/usr/bin/env python3
"""Launch label-free Mem-JEPA Stage-C age jobs on GPUs 0, 1, and 2."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/mem_jepa_stage_c"


def python_bin() -> str:
    candidate = ROOT / ".venv/bin/python"
    return str(candidate if candidate.exists() else Path(sys.executable))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--slots", type=int, default=6)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=9600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    log_dir = output / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(4, 0), (8, 1), (15, 2)]
    processes: list[tuple[int, int, subprocess.Popen[bytes], object]] = []
    for age, gpu in jobs:
        log = (log_dir / f"age_{age}.log").open("wb")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        command = [
            python_bin(),
            str(ROOT / "scripts/run_mem_jepa_stage_c.py"),
            "--age", str(age),
            "--output", str(output),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--eval-batch-size", str(args.eval_batch_size),
            "--dim", str(args.dim),
            "--slots", str(args.slots),
            "--heads", str(args.heads),
            "--temperature", str(args.temperature),
            "--seed", str(args.seed + age),
            "--device", "cuda:0",
        ]
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        processes.append((age, gpu, process, log))
        print(f"[stage-c-launch] age={age} gpu={gpu} pid={process.pid}", flush=True)
    failed = []
    try:
        while processes:
            still_running = []
            for age, gpu, process, log in processes:
                code = process.poll()
                if code is None:
                    still_running.append((age, gpu, process, log))
                    continue
                log.close()
                print(f"[stage-c-done] age={age} gpu={gpu} code={code}", flush=True)
                if code != 0:
                    failed.append((age, code))
            processes = still_running
            if processes:
                time.sleep(15)
    finally:
        for _, _, process, log in processes:
            if process.poll() is None:
                process.terminate()
            log.close()
    if failed:
        raise SystemExit(f"Stage-C jobs failed: {failed}")
    subprocess.check_call([
        python_bin(),
        str(ROOT / "scripts/run_mem_jepa_stage_c.py"),
        "--aggregate",
        "--output", str(output),
    ], cwd=ROOT)


if __name__ == "__main__":
    main()
