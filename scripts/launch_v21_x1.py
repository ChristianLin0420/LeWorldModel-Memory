#!/usr/bin/env python3
"""Launch the V21 X1 preregistered confirmation wave (docs/V21_PROPOSAL.md
4/X1; registration frozen in outputs/v21_x1/registration.json BEFORE the
X0b sweep ran).

Grid: {lkc_rfix @ its registered V19 recipe (lr 3e-4)} x {envelope* from the
X0b selection, with its swept recipe} x {t1, t3, t4} x FRESH seeds 10-19 =
60 runs on the vicreg host, 2/GPU.  Then the categorical probe battery
(t1/t3) and the frozen gate (scripts/gates_v21_x1.py), which computes t4
itself under the repaired probe family.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.train_v21_x0 import arm_lr

GPUS = (0, 1, 2)
TASKS = ("t1", "t3", "t4")
FRESH_SEEDS = tuple(range(10, 20))


@dataclass
class Job:
    task: str
    arm: str
    seed: int

    @property
    def name(self) -> str:
        return f"{self.task}_{self.arm}_s{self.seed}"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/v21_x1")
    parser.add_argument("--sweep-summary",
                        default="outputs/v21_x0/sweep_summary.json")
    parser.add_argument("--jobs-per-gpu", type=int, default=2)
    parser.add_argument("--project", default="lewm-v21")
    parser.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    parser.add_argument("--no-wandb", dest="wandb", action="store_false")
    return parser.parse_args(argv)


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


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    registration = Path("outputs/v21_x1/registration.json")
    if not registration.exists():
        raise SystemExit("X1 registration missing")
    envelope = json.loads(Path(args.sweep_summary).read_text())["envelope_star"]
    print(f"[v21-x1] envelope* = {envelope} (lr {arm_lr(envelope)}) | "
          f"lkc_rfix @ registered lr 3e-4 | fresh seeds "
          f"{FRESH_SEEDS[0]}-{FRESH_SEEDS[-1]}", flush=True)
    log_dir = Path(args.output) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    queue = [Job(task, arm, seed) for task, arm, seed in
             product(TASKS, ("lkc_rfix", envelope), FRESH_SEEDS)]
    queue = [job for job in queue
             if not (Path(args.output) / job.task / job.arm / f"s{job.seed}"
                     / "gates.json").is_file()]
    print(f"[v21-x1] {len(queue)} jobs ({args.jobs_per_gpu}/GPU)", flush=True)

    running: list[tuple[subprocess.Popen, Job, int]] = []
    gpu_load = {gpu: 0 for gpu in GPUS}
    done = crashed = 0
    while queue or running:
        while queue and min(gpu_load.values()) < args.jobs_per_gpu:
            gpu = min(GPUS, key=gpu_load.__getitem__)
            job = queue.pop(0)
            command = [
                sys.executable, str(REPO / "scripts" / "train_v21_x0.py"),
                "--task", job.task, "--host", "vicreg", "--arm", job.arm,
                "--seed", str(job.seed), "--output", args.output,
                "--lr", str(arm_lr(job.arm)),
                "--wandb" if args.wandb else "--no-wandb",
                "--wandb-project", args.project,
            ]
            log = open(log_dir / f"{job.name}.log", "w")
            running.append((subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT,
                env=_job_env(gpu, args.wandb), cwd=REPO), job, gpu))
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
                print(f"[v21-x1] CRASH {job.name} (exit {code})", flush=True)
                time.sleep(30)
        running = still
        tags = ",".join(f"{j.name}@gpu{g}" for _, j, g in running)
        print(f"\r[v21-x1] {done} done, {crashed} crashed | running: {tags}",
              end="", flush=True)
    print(f"\n[v21-x1] wave finished: {done} ok, {crashed} crashed", flush=True)

    env = _job_env(GPUS[0], wandb_on=False)
    import scripts.eval_v19_p2 as p2eval
    for export_path in p2eval.discover_exports(args.output):
        if (export_path.parent / p2eval.RESULTS_NAME).exists():
            continue
        if "/t4/" in str(export_path):
            continue          # t4 is scored by the gate's repaired family
        results = p2eval.process_run(export_path)
        print(f"[v21-x1] probes {export_path.parent}: "
              f"{results['registered']['mean']:.3f}", flush=True)

    subprocess.run([sys.executable, str(REPO / "scripts" / "gates_v21_x1.py"),
                    "--root", args.output, "--envelope", envelope],
                   cwd=REPO, env=env, check=False)


if __name__ == "__main__":
    main()
