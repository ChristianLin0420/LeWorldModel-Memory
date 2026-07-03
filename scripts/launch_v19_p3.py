#!/usr/bin/env python3
"""Launch the V19 P3 frozen confirmation grid (docs/V19_PROPOSAL.md 6, 10).

Frozen confirmation set (section 10 freeze event): tasks T1/T3/T4 x the FULL
section-4.3 trained arm deck (two references + no-carrier, both LKC
candidates, and the six single-knob interventions) x ``--seeds`` training
seeds (default 3 — set this from the P2 power analysis at launch time,
proposal 4.5 Tier 1), on the VICReg host (the P0 fallback).  The trainer is
scripts/train_v19_p2.py verbatim — the P3 grid is the same trainer on the
confirmation tasks with the full deck.

Data root: the amendment-2 caches (``outputs/v19_p0_a2/data``).  The
pre-amendment banks under ``outputs/v19_p0/data`` still carry valid sha
sidecars, so pointing the trainer at the old default would silently train on
pre-salience tasks — the registered default here is therefore the a2 root.

After the grid, the chain runs in gate order:
  1. scripts/eval_v19_p2.py       — probe_results.json per cell;
  2. scripts/counterfactual_v19.py — action-swap counterfactuals on the
     lkc / lkc_b0 cells (Tier-2 transport gate input);
  3. scripts/gates_v19_p3.py       — the three-tier evaluator
     (p3_gates.json / p3_gates.md);
  4. scripts/aggregate_v19_p2.py   — descriptive arm summary.

Resumable: cells whose ``gates.json`` already exists are skipped (the P2
convention); partially written cells must be cleaned by hand — the trainer
refuses to overwrite its own artifacts.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parents[1]
GPUS = (0, 1, 2)

TASKS = ("t1", "t3", "t4")
ARMS = ("none", "acgru", "acssm", "lkc", "lkc_nll", "lkc_k0", "lkc_b0",
        "lkc_kfix", "lkc_rfix", "lkc_alearn", "lkc_a2")
COUNTERFACTUAL_ARMS = ("lkc", "lkc_b0")
DEFAULT_P0_DATA_ROOT = "outputs/v19_p0_a2/data"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/v19_p3")
    parser.add_argument("--host", default="vicreg", choices=("sigreg", "vicreg"),
                        help="P0-selected host (fallback rule, proposal 4.1; "
                             "the section-10 freeze fixes vicreg)")
    parser.add_argument("--seeds", type=int, default=3,
                        help="training seeds per cell — set from the P2 "
                             "power analysis at launch time")
    parser.add_argument("--jobs-per-gpu", type=int, default=3,
                        help="concurrency per GPU (P2 measured footprint "
                             "with need_weights=False attention)")
    parser.add_argument("--p0-data-root", default=DEFAULT_P0_DATA_ROOT,
                        help="amendment-2 cache root; NEVER the pre-a2 "
                             "outputs/v19_p0/data (stale banks validate)")
    parser.add_argument("--project", default="lewm-v19")
    parser.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    parser.add_argument("--no-wandb", dest="wandb", action="store_false")
    return parser.parse_args(argv)


@dataclass
class Job:
    task: str
    arm: str
    seed: int

    @property
    def name(self) -> str:
        return f"{self.task}_{self.arm}_s{self.seed}"


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


def _launch(job: Job, gpu: int, args: argparse.Namespace, log_dir: Path
            ) -> subprocess.Popen:
    command = [
        sys.executable, str(REPO / "scripts" / "train_v19_p2.py"),
        "--task", job.task, "--host", args.host, "--arm", job.arm,
        "--seed", str(job.seed), "--output", args.output,
        "--p0-data-root", args.p0_data_root,
        "--wandb" if args.wandb else "--no-wandb",
        "--wandb-project", args.project,
    ]
    log = open(log_dir / f"{job.name}.log", "w")
    return subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                            env=_job_env(gpu, args.wandb), cwd=REPO)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.seeds < 1:
        raise ValueError("--seeds must be >= 1")
    seeds = tuple(range(args.seeds))
    log_dir = Path(args.output) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    queue = [Job(t, a, s) for t, a, s in product(TASKS, ARMS, seeds)]
    queue = [job for job in queue
             if not (Path(args.output) / job.task / job.arm / f"s{job.seed}"
                     / "gates.json").is_file()]
    print(f"[v19-p3] {len(queue)} jobs to run "
          f"({args.jobs_per_gpu}/GPU over GPUs {GPUS}; grid "
          f"{len(TASKS)}x{len(ARMS)}x{len(seeds)})")

    running: list[tuple[subprocess.Popen, Job, int]] = []
    gpu_load = {gpu: 0 for gpu in GPUS}
    done = crashed = 0
    while queue or running:
        while queue and min(gpu_load.values()) < args.jobs_per_gpu:
            gpu = min(GPUS, key=gpu_load.__getitem__)
            job = queue.pop(0)
            running.append((_launch(job, gpu, args, log_dir), job, gpu))
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
                print(f"[v19-p3] CRASH {job.name} (exit {code})")
        running = still
        tags = ",".join(f"{j.name}@gpu{g}" for _, j, g in running)
        print(f"\r[v19-p3] {done} done, {crashed} crashed | running: {tags}",
              end="", flush=True)
    print(f"\n[v19-p3] grid finished: {done} ok, {crashed} crashed")

    # Post-grid chain (gate order: probes -> counterfactual -> tiers -> agg).
    chain_env = _job_env(GPUS[0], args.wandb)
    chain = [
        [sys.executable, str(REPO / "scripts" / "eval_v19_p2.py"),
         "--root", args.output],
        [sys.executable, str(REPO / "scripts" / "counterfactual_v19.py"),
         "--root", args.output, "--arms", ",".join(COUNTERFACTUAL_ARMS),
         "--tasks", ",".join(TASKS),
         "--wandb" if args.wandb else "--no-wandb",
         "--wandb-project", args.project],
        [sys.executable, str(REPO / "scripts" / "gates_v19_p3.py"),
         "--root", args.output, "--p0-data-root", args.p0_data_root,
         "--seeds", ",".join(str(seed) for seed in seeds)],
        [sys.executable, str(REPO / "scripts" / "aggregate_v19_p2.py"),
         "--root", args.output],
    ]
    for command in chain:
        subprocess.run(command, cwd=REPO, env=chain_env, check=False)


if __name__ == "__main__":
    main()
