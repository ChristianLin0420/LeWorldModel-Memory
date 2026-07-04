#!/usr/bin/env python3
"""Launch the V20 W1 development grid (docs/V20_PROPOSAL.md 6).

Host selection is read from the W0 summary (fail-closed rule): the visreg
lambda* arm if claim 1 passed, else the vicreg fallback.  Grid: 3 trained
arms (lkc_rfix / acgru / none — the DFC family shares the lkc_rfix
checkpoint) x {t1dev, t3dev} x 3 seeds = 18 runs, 2 jobs/GPU over GPUs 0-2
(carrier-run sizing, the V19 P2 lesson), serial cache pre-generation, 30 s
crash cooldown.  Then scripts/eval_v20_w1.py (DFC deployment variants + the
probe battery) and scripts/aggregate_v20_w1.py (claims 4-5 dev gates, rho*,
the W3 power analysis).
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

GPUS = (0, 1, 2)
SEEDS = (0, 1, 2)
ARMS = ("lkc_rfix", "acgru", "none")
TASKS = ("t1dev", "t3dev")


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
    parser.add_argument("--output", default="outputs/v20_w1")
    parser.add_argument("--w0-summary", default="outputs/v20_w0/w0_summary.json")
    parser.add_argument("--host", default=None,
                        help="override the W0-selected host")
    parser.add_argument("--p2-data-root", default="outputs/v19_p2/data")
    parser.add_argument("--jobs-per-gpu", type=int, default=2)
    parser.add_argument("--project", default="lewm-v20")
    parser.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    parser.add_argument("--no-wandb", dest="wandb", action="store_false")
    return parser.parse_args(argv)


def select_host(args: argparse.Namespace) -> str:
    if args.host:
        return args.host
    summary_path = Path(args.w0_summary)
    if not summary_path.exists():
        raise SystemExit(f"missing W0 summary {summary_path}; run W0 first "
                         f"or pass --host explicitly")
    summary = json.loads(summary_path.read_text())
    if summary.get("claim1_visreg_host_healthy") and summary.get("lambda_star"):
        return summary["lambda_star"]
    return "vicreg"


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


def _launch(job: Job, gpu: int, host: str, args: argparse.Namespace,
            log_dir: Path) -> subprocess.Popen:
    command = [
        sys.executable, str(REPO / "scripts" / "train_v20_w1.py"),
        "--task", job.task, "--host", host, "--arm", job.arm,
        "--seed", str(job.seed), "--output", args.output,
        "--p2-data-root", args.p2_data_root,
        "--wandb" if args.wandb else "--no-wandb",
        "--wandb-project", args.project,
    ]
    log = open(log_dir / f"{job.name}.log", "w")
    return subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                            env=_job_env(gpu, args.wandb), cwd=REPO)


def _pregenerate_caches(args: argparse.Namespace, log_dir: Path) -> None:
    for task in TASKS:
        command = [sys.executable, "-c",
                   "import sys; sys.path.insert(0, 'scripts'); "
                   "import train_v20_w1 as w1; "
                   f"w1.resolve_banks({task!r}, {args.p2_data_root!r}, "
                   f"{args.output!r} + '/data')"]
        log = open(log_dir / f"data_{task}.log", "w")
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                env=_job_env(GPUS[0], wandb_on=False), cwd=REPO)
        if result.returncode != 0:
            raise SystemExit(f"cache generation failed for {task}; "
                             f"see {log_dir}/data_{task}.log")
        print(f"[v20-w1] cache ready: {task}", flush=True)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    host = select_host(args)
    print(f"[v20-w1] host = {host}", flush=True)
    log_dir = Path(args.output) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (Path(args.output) / "host_selection.json").write_text(json.dumps(
        {"host": host, "source": args.w0_summary if not args.host
         else "override"}, indent=2) + "\n")

    _pregenerate_caches(args, log_dir)

    queue = [Job(task, arm, seed)
             for task, arm, seed in product(TASKS, ARMS, SEEDS)]
    queue = [job for job in queue
             if not (Path(args.output) / job.task / job.arm / f"s{job.seed}"
                     / "gates.json").is_file()]
    print(f"[v20-w1] {len(queue)} jobs "
          f"({args.jobs_per_gpu}/GPU over GPUs {GPUS})", flush=True)

    running: list[tuple[subprocess.Popen, Job, int]] = []
    gpu_load = {gpu: 0 for gpu in GPUS}
    done = crashed = 0
    while queue or running:
        while queue and min(gpu_load.values()) < args.jobs_per_gpu:
            gpu = min(GPUS, key=gpu_load.__getitem__)
            job = queue.pop(0)
            running.append((_launch(job, gpu, host, args, log_dir), job, gpu))
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
                print(f"[v20-w1] CRASH {job.name} (exit {code})", flush=True)
                time.sleep(30)
        running = still
        tags = ",".join(f"{j.name}@gpu{g}" for _, j, g in running)
        print(f"\r[v20-w1] {done} done, {crashed} crashed | running: {tags}",
              end="", flush=True)
    print(f"\n[v20-w1] grid finished: {done} ok, {crashed} crashed", flush=True)

    evaluate = [sys.executable, str(REPO / "scripts" / "eval_v20_w1.py"),
                "--root", args.output, "--p2-data-root", args.p2_data_root]
    aggregate = [sys.executable, str(REPO / "scripts" / "aggregate_v20_w1.py"),
                 "--root", args.output]
    subprocess.run(evaluate, cwd=REPO, check=False,
                   env=_job_env(GPUS[0], wandb_on=False))
    subprocess.run(aggregate, cwd=REPO, check=False)


if __name__ == "__main__":
    main()
