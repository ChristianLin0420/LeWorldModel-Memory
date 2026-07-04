#!/usr/bin/env python3
"""Launch the V20 W3 frozen confirmation (docs/V20_PROPOSAL.md 6).

Everything is frozen before launch: host from the W0 fail-closed rule,
rho*/eta* from the W1 summary, n seeds from the W1 power analysis
(registered rule: the smallest n with >= 80% power for the registered +5%
effect, clamped to [5, 10]; missing/None => 10).

Chain: train {lkc_rfix, acgru, none} x {t1, t3, t4} x n seeds (2/GPU, crash
cooldown, resumable) -> stationary DFC variants + probe battery
(scripts/eval_v20_w1.py with the two frozen variants only) -> drift protocol
on the categorical frozen tasks (scripts/eval_v20_w2.py into <output>/drift,
canonical arm names) -> drift aggregation -> confirmatory gates
(scripts/gates_v20_w3.py: crossed bootstrap + Holm).
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
ARMS = ("lkc_rfix", "acgru", "none")
TASKS = ("t1", "t3", "t4")
DRIFT_TASKS = "t1,t3"


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
    parser.add_argument("--output", default="outputs/v20_w3")
    parser.add_argument("--w0-summary", default="outputs/v20_w0/w0_summary.json")
    parser.add_argument("--w1-summary", default="outputs/v20_w1/w1_summary.json")
    parser.add_argument("--jobs-per-gpu", type=int, default=2)
    parser.add_argument("--project", default="lewm-v20")
    parser.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    parser.add_argument("--no-wandb", dest="wandb", action="store_false")
    return parser.parse_args(argv)


def frozen_parameters(args: argparse.Namespace) -> dict:
    w0 = json.loads(Path(args.w0_summary).read_text())
    w1 = json.loads(Path(args.w1_summary).read_text())
    host = (w0["lambda_star"]
            if w0.get("claim1_visreg_host_healthy") and w0.get("lambda_star")
            else "vicreg")
    power = (w1.get("w3_power_analysis") or {})
    n = None
    if power.get("status") == "ok":
        n = (power["effects"].get("registered_plus_5pct") or {}).get(
            "smallest_n_with_80pct_power")
    n = max(5, min(10, n)) if n else 10
    if not w1.get("rho_star") or not w1.get("eta_star"):
        raise SystemExit("W1 summary lacks rho*/eta*; W3 cannot freeze")
    return {"host": host, "n_seeds": n,
            "rho_star": w1["rho_star"], "eta_star": w1["eta_star"],
            "subsumption_dev_pass": (w1.get("claim4_subsumption") or {}
                                     ).get("pass")}


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
        "--wandb" if args.wandb else "--no-wandb",
        "--wandb-project", args.project,
    ]
    log = open(log_dir / f"{job.name}.log", "w")
    return subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                            env=_job_env(gpu, args.wandb), cwd=REPO)


def run_grid(queue: list[Job], host: str, args: argparse.Namespace,
             log_dir: Path) -> None:
    queue = [job for job in queue
             if not (Path(args.output) / job.task / job.arm / f"s{job.seed}"
                     / "gates.json").is_file()]
    print(f"[v20-w3] {len(queue)} training jobs "
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
                print(f"[v20-w3] CRASH {job.name} (exit {code})", flush=True)
                time.sleep(30)
        running = still
        tags = ",".join(f"{j.name}@gpu{g}" for _, j, g in running)
        print(f"\r[v20-w3] {done} done, {crashed} crashed | running: {tags}",
              end="", flush=True)
    print(f"\n[v20-w3] grid finished: {done} ok, {crashed} crashed", flush=True)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    frozen = frozen_parameters(args)
    seeds = list(range(frozen["n_seeds"]))
    seed_list = ",".join(map(str, seeds))
    print(f"[v20-w3] FROZEN: host={frozen['host']} n={frozen['n_seeds']} "
          f"dfc={frozen['rho_star']} etafix={frozen['eta_star']}", flush=True)
    log_dir = Path(args.output) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (Path(args.output) / "w3_frozen.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n")

    queue = [Job(task, arm, seed)
             for task, arm, seed in product(TASKS, ARMS, seeds)]
    run_grid(queue, frozen["host"], args, log_dir)

    env = _job_env(GPUS[0], wandb_on=False)
    steps = [
        ("stationary-eval",
         [sys.executable, str(REPO / "scripts" / "eval_v20_w1.py"),
          "--root", args.output, "--tasks", ",".join(TASKS),
          "--seeds", seed_list,
          "--variants", f"{frozen['rho_star']},{frozen['eta_star']}"]),
        ("drift-eval",
         [sys.executable, str(REPO / "scripts" / "eval_v20_w2.py"),
          "--output", str(Path(args.output) / "drift"),
          "--w1-root", args.output, "--w1-summary", args.w1_summary,
          "--tasks", DRIFT_TASKS, "--seeds", seed_list]),
        ("drift-aggregate",
         [sys.executable, str(REPO / "scripts" / "aggregate_v20_w2.py"),
          "--root", str(Path(args.output) / "drift"),
          "--tasks", DRIFT_TASKS, "--seeds", seed_list]),
        ("gates",
         [sys.executable, str(REPO / "scripts" / "gates_v20_w3.py"),
          "--root", args.output, "--w1-summary", args.w1_summary,
          "--seeds", seed_list]),
    ]
    for name, command in steps:
        print(f"[v20-w3] step: {name}", flush=True)
        result = subprocess.run(command, cwd=REPO, env=env, check=False)
        if result.returncode != 0:
            print(f"[v20-w3] step {name} FAILED (exit {result.returncode}) — "
                  f"chain stopped for inspection", flush=True)
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
