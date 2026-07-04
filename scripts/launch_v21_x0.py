#!/usr/bin/env python3
"""Launch the V21 X0b baseline-parity sweep (docs/V21_PROPOSAL.md 4/X0.3).

Registered grid on the dev tasks (vicreg host, V19 dev banks read-only):

  acgru width {64, 102, 160} x lr {1e-4, 3e-4, 1e-3}   9 configs
  acgru_chrono (matched width) x lr sweep               3
  gdelta (parameter-matched) x lr sweep                 3
  acssm (V19 recipe, reinstated)                        1

16 configs x {t1dev, t3dev} x seeds {0, 1} = 64 runs, 2/GPU over GPUs 0-2,
then the probe battery (scripts/eval_v19_p2.py machinery via
scripts/eval_v20_w1.py's probe loop) and the envelope* selection
(max pooled dev mean over the registered probe — the rule frozen in
outputs/v21_x1/registration.json BEFORE this launcher ran).

Writes outputs/v21_x0/sweep_summary.{json,md} with envelope*.
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

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.train_v21_x0 import X0_ARMS, arm_lr

GPUS = (0, 1, 2)
TASKS = ("t1dev", "t3dev")
SEEDS = (0, 1)


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
    parser.add_argument("--output", default="outputs/v21_x0")
    parser.add_argument("--p2-data-root", default="outputs/v19_p2/data")
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


def _launch(job: Job, gpu: int, args: argparse.Namespace, log_dir: Path
            ) -> subprocess.Popen:
    command = [
        sys.executable, str(REPO / "scripts" / "train_v21_x0.py"),
        "--task", job.task, "--host", "vicreg", "--arm", job.arm,
        "--seed", str(job.seed), "--output", args.output,
        "--p2-data-root", args.p2_data_root,
        "--lr", str(arm_lr(job.arm)),
        "--wandb" if args.wandb else "--no-wandb",
        "--wandb-project", args.project,
    ]
    log = open(log_dir / f"{job.name}.log", "w")
    return subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                            env=_job_env(gpu, args.wandb), cwd=REPO)


def run_grid(args: argparse.Namespace, log_dir: Path) -> None:
    queue = [Job(task, arm, seed)
             for task, arm, seed in product(TASKS, X0_ARMS, SEEDS)]
    queue = [job for job in queue
             if not (Path(args.output) / job.task / job.arm / f"s{job.seed}"
                     / "gates.json").is_file()]
    print(f"[v21-x0] {len(queue)} sweep jobs "
          f"({args.jobs_per_gpu}/GPU over GPUs {GPUS})", flush=True)
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
                print(f"[v21-x0] CRASH {job.name} (exit {code})", flush=True)
                time.sleep(30)
        running = still
        tags = ",".join(f"{j.name}@gpu{g}" for _, j, g in running)
        print(f"\r[v21-x0] {done} done, {crashed} crashed | running: {tags}",
              end="", flush=True)
    print(f"\n[v21-x0] sweep finished: {done} ok, {crashed} crashed",
          flush=True)


def run_probes(args: argparse.Namespace) -> None:
    import scripts.eval_v19_p2 as p2eval
    exports = p2eval.discover_exports(args.output)
    print(f"[v21-x0] probing {len(exports)} exports", flush=True)
    for export_path in exports:
        if (export_path.parent / p2eval.RESULTS_NAME).exists():
            continue
        results = p2eval.process_run(export_path)
        print(f"[v21-x0] {export_path.parent.relative_to(args.output)}: "
              f"registered={results['registered']['mean']:.3f}", flush=True)


def select_envelope(args: argparse.Namespace) -> None:
    root = Path(args.output)
    table = {}
    for arm in X0_ARMS:
        scores = []
        for task in TASKS:
            for seed in SEEDS:
                path = root / task / arm / f"s{seed}" / "probe_results.json"
                if path.exists():
                    scores.append(float(json.loads(path.read_text())
                                        ["registered"]["mean"]))
        if len(scores) == len(TASKS) * len(SEEDS):
            table[arm] = {"pooled_mean": float(np.mean(scores)),
                          "scores": [round(s, 4) for s in scores]}
    if not table:
        raise SystemExit("no complete sweep cells — selection impossible")
    envelope = max(table, key=lambda arm: table[arm]["pooled_mean"])
    summary = {
        "schema_version": 1,
        "study": "v21-x0b-baseline-parity-sweep",
        "tasks": list(TASKS),
        "seeds": list(SEEDS),
        "configs": table,
        "envelope_star": envelope,
        "envelope_star_lr": arm_lr(envelope),
        "selection_rule": "max pooled dev mean (frozen in "
                          "outputs/v21_x1/registration.json)",
    }
    (root / "sweep_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    lines = ["# V21 X0b — baseline-parity sweep", "",
             "| config | pooled dev mean | scores (t1dev s0/s1, t3dev s0/s1) |",
             "|---|---|---|"]
    for arm, row in sorted(table.items(),
                           key=lambda item: -item[1]["pooled_mean"]):
        marker = " **← envelope\\***" if arm == envelope else ""
        lines.append(f"| {arm} | {row['pooled_mean']:.4f} | "
                     f"{row['scores']} |{marker}")
    (root / "sweep_summary.md").write_text("\n".join(lines) + "\n")
    print(f"[v21-x0] envelope* = {envelope} "
          f"(pooled {table[envelope]['pooled_mean']:.4f})", flush=True)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    registration = REPO / "outputs" / "v21_x1" / "registration.json"
    if not registration.exists():
        raise SystemExit("X1 registration missing — freeze the gate before "
                         "running the sweep (scripts/gates_v21_x1.py "
                         "--register)")
    log_dir = Path(args.output) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_grid(args, log_dir)
    run_probes(args)
    select_envelope(args)


if __name__ == "__main__":
    main()
