#!/usr/bin/env python3
"""Launch Stage-E executed-use seed jobs on GPUs 0, 1, and 2."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/mem_jepa_stage_e"
SEEDS = (0, 1, 2, 3, 4)


def python_bin() -> str:
    candidate = ROOT / ".venv/bin/python"
    return str(candidate if candidate.exists() else Path(sys.executable))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--slots", type=int, default=6)
    parser.add_argument("--heads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    log_dir = output / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    queue = list(SEEDS)
    gpus = [0, 1, 2]
    running: dict[int, tuple[int, subprocess.Popen[bytes], object]] = {}
    failed: list[tuple[int, int]] = []
    while queue or running:
        for gpu in gpus:
            if gpu in running or not queue:
                continue
            seed = queue.pop(0)
            log = (log_dir / f"seed_{seed}.log").open("wb")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            command = [
                python_bin(),
                str(ROOT / "scripts/run_mem_jepa_stage_e.py"),
                "--seed-index", str(seed),
                "--output", str(output),
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--eval-batch-size", str(args.eval_batch_size),
                "--dim", str(args.dim),
                "--slots", str(args.slots),
                "--heads", str(args.heads),
                "--device", "cuda:0",
            ]
            process = subprocess.Popen(
                command, cwd=ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT)
            running[gpu] = (seed, process, log)
            print(f"[stage-e-launch] gpu={gpu} pid={process.pid} seed={seed}", flush=True)
        time.sleep(10)
        for gpu, (seed, process, log) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            print(f"[stage-e-done] gpu={gpu} code={code} seed={seed}", flush=True)
            if code != 0:
                failed.append((seed, int(code)))
            del running[gpu]
        if failed:
            for _, process, log in running.values():
                if process.poll() is None:
                    process.terminate()
                log.close()
            raise SystemExit(f"Stage-E jobs failed: {failed}")
    subprocess.check_call([
        python_bin(),
        str(ROOT / "scripts/run_mem_jepa_stage_e.py"),
        "--aggregate",
        "--output", str(output),
        "--draws", str(args.draws),
    ], cwd=ROOT)


if __name__ == "__main__":
    main()
