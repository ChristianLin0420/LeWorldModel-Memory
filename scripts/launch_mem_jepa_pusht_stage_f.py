#!/usr/bin/env python3
"""Launch PushT Stage-F Mem-JEPA cells across GPUs 0/1/2."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_mem_jepa_pusht_stage_f.py"
DEFAULT_OUTPUT = ROOT / "outputs/mem_jepa_pusht_stage_f"


def python_bin() -> str:
    candidate = ROOT / ".venv/bin/python"
    return str(candidate if candidate.exists() else Path(sys.executable))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--gpus", nargs="*", default=["0", "1", "2"])
    parser.add_argument("--tasks", nargs="*", default=[
        "transient-visual-token-recall",
        "multi-item-visual-binding-recall",
    ])
    parser.add_argument("--ages", type=int, nargs="*", default=[4, 8, 15])
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--target-mode", default="delta_compact",
                        choices=["cue_compact", "delta_compact",
                                 "binding_slots", "delta_binding_slots",
                                 "cue_binding_slots"])
    parser.add_argument("--negative-mode", default="batch_roll",
                        choices=["batch_roll", "binding_permutation"])
    parser.add_argument("--adapter", default="residual",
                        choices=["residual", "host_writer"])
    parser.add_argument("--host-weight", type=float, default=1.0)
    parser.add_argument("--context-weight", type=float, default=1.0)
    parser.add_argument("--memory-weight", type=float, default=1.0)
    parser.add_argument("--residual-l2-weight", type=float, default=1.0e-4)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def result_path(output: Path, task: str, seed: int, age: int) -> Path:
    return output / task / f"s{seed}" / f"age_{age}" / "result.json"


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    jobs = [
        (task, age, seed)
        for task in args.tasks
        for age in args.ages
        for seed in args.seeds
        if not result_path(output, task, seed, age).is_file()
    ]
    print(f"[launcher] pending jobs={len(jobs)}", flush=True)
    if args.dry_run:
        for task, age, seed in jobs:
            print(f"{task} age={age} seed={seed}")
        return
    running: dict[str, tuple[subprocess.Popen[bytes], tuple[str, int, int], Path]] = {}
    completed = 0
    failed = 0
    while jobs or running:
        for gpu in args.gpus:
            if gpu in running or not jobs:
                continue
            task, age, seed = jobs.pop(0)
            log_path = logs / f"{task}_age{age}_s{seed}_gpu{gpu}.log"
            cmd = [
                python_bin(), str(RUNNER),
                "--output", str(output),
                "--task", task,
                "--age", str(age),
                "--seed", str(seed),
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--eval-batch-size", str(args.eval_batch_size),
                "--candidate-count", str(args.candidate_count),
                "--target-mode", args.target_mode,
                "--negative-mode", args.negative_mode,
                "--adapter", args.adapter,
                "--host-weight", str(args.host_weight),
                "--context-weight", str(args.context_weight),
                "--memory-weight", str(args.memory_weight),
                "--residual-l2-weight", str(args.residual_l2_weight),
                "--device", "cuda:0",
            ]
            if args.diagnostics:
                cmd.append("--diagnostics")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            stream = log_path.open("wb")
            process = subprocess.Popen(
                cmd, cwd=str(ROOT), env=env, stdout=stream,
                stderr=subprocess.STDOUT)
            running[gpu] = (process, (task, age, seed), log_path)
            print(f"[launcher] started gpu={gpu} {task} age={age} seed={seed} "
                  f"log={log_path}", flush=True)
        time.sleep(float(args.poll_seconds))
        for gpu, (process, job, log_path) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            task, age, seed = job
            del running[gpu]
            if code == 0:
                completed += 1
                print(f"[launcher] completed gpu={gpu} {task} age={age} "
                      f"seed={seed}", flush=True)
            else:
                failed += 1
                print(f"[launcher] failed gpu={gpu} {task} age={age} "
                      f"seed={seed} code={code} log={log_path}", flush=True)
                jobs.clear()
                for other_gpu, (other, _, _) in list(running.items()):
                    other.terminate()
                    print(f"[launcher] terminated gpu={other_gpu}", flush=True)
                raise SystemExit(code)
    print(f"[launcher] done completed={completed} failed={failed}", flush=True)
    aggregate_cmd = [
        python_bin(), str(RUNNER), "--output", str(output), "--aggregate",
        "--target-mode", args.target_mode,
        "--negative-mode", args.negative_mode,
        "--adapter", args.adapter,
        "--tasks", *args.tasks,
        "--ages", *[str(value) for value in args.ages],
        "--seeds", *[str(value) for value in args.seeds],
    ]
    subprocess.check_call(aggregate_cmd, cwd=str(ROOT))


if __name__ == "__main__":
    main()
