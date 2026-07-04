#!/usr/bin/env python3
"""Launch the V20 W0 host preflight + salience ladder (docs/V20_PROPOSAL.md 6).

Registered grid, executed in two training waves plus certificates:

- Wave 1: visreg lambda sweep on t1 (visreg60/75/90 x 3 seeds = 9) and the
  vicreg reference on the ladder levels (t1s1/t1s2/t1s3 x 3 seeds = 9).
- lambda* selection (aggregate_v20_w0.select_lambda_star): all-3-seed health
  pass on t1, then max mean final effective rank.  No passing lambda =>
  claim 1 falsified fail-closed: the visreg wave 2 is skipped and the vicreg
  fallback stands for W1.
- Wave 2 (lambda* only): visreg lambda* on t3/t4 (6) and on the ladder (9).
- Certificates: scripts/certify_v20_w0.py on the full ladder for vicreg and
  lambda*, parallel over GPUs by task.
- Aggregation: scripts/aggregate_v20_w0.py (claims 1-2, s*).

Scheduling follows the V19 P2 lessons: serial cache pre-generation before any
training job (no bank races), 3 jobs/GPU on GPUs 0-2, and a 30 s cooldown
after any crash before the slot is refilled.

vicreg t1 cells are NOT trained here: the frozen P0-a2 encoders are reused as
the amendment-2 reference — this launcher copies nothing and reads them only
at certificate time via --vicreg-t1-root.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import scripts.aggregate_v20_w0 as w0_aggregate

GPUS = (0, 1, 2)
SEEDS = (0, 1, 2)
VISREG_ARMS = ("visreg60", "visreg75", "visreg90")
LADDER_NEW = ("t1s1", "t1s2", "t1s3")
LADDER_ALL = ("t1s1", "t1s2", "t1s3", "t1")
P0A2_ROOT = "outputs/v19_p0_a2"


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
    parser.add_argument("--output", default="outputs/v20_w0")
    parser.add_argument("--p0-data-root", default=f"{P0A2_ROOT}/data")
    parser.add_argument("--vicreg-t1-root", default=P0A2_ROOT,
                        help="frozen P0-a2 root holding vicreg/t1 encoders")
    parser.add_argument("--jobs-per-gpu", type=int, default=3)
    parser.add_argument("--project", default="lewm-v20")
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
        sys.executable, str(REPO / "scripts" / "train_v20_w0.py"),
        "--task", job.task, "--arm", job.arm, "--seed", str(job.seed),
        "--output", args.output, "--p0-data-root", args.p0_data_root,
        "--wandb" if args.wandb else "--no-wandb",
        "--wandb-project", args.project,
    ]
    log = open(log_dir / f"{job.name}.log", "w")
    return subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                            env=_job_env(gpu, args.wandb), cwd=REPO)


def _pregenerate_caches(args: argparse.Namespace, log_dir: Path) -> None:
    """Serial per-task bank generation (the V19 P2 race lesson)."""
    for task in LADDER_NEW:
        command = [sys.executable, "-c",
                   "import sys; sys.path.insert(0, 'scripts'); "
                   "import train_v20_w0 as w0; "
                   f"w0.resolve_banks({task!r}, {args.p0_data_root!r}, "
                   f"{args.output!r} + '/data')"]
        log = open(log_dir / f"data_{task}.log", "w")
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                env=_job_env(GPUS[0], wandb_on=False), cwd=REPO)
        if result.returncode != 0:
            raise SystemExit(f"cache generation failed for {task}; "
                             f"see {log_dir}/data_{task}.log")
        print(f"[v20-w0] cache ready: {task}", flush=True)


def _run_wave(queue: list[Job], args: argparse.Namespace, log_dir: Path,
              wave: str) -> tuple[int, int]:
    queue = [job for job in queue
             if not (Path(args.output) / job.task / job.arm / f"s{job.seed}"
                     / "gates.json").is_file()]
    print(f"[v20-w0] {wave}: {len(queue)} jobs "
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
                print(f"[v20-w0] CRASH {job.name} (exit {code})", flush=True)
                time.sleep(30)   # let the dying process release GPU memory
        running = still
        tags = ",".join(f"{j.name}@gpu{g}" for _, j, g in running)
        print(f"\r[v20-w0] {wave}: {done} done, {crashed} crashed | "
              f"running: {tags}", end="", flush=True)
    print(f"\n[v20-w0] {wave} finished: {done} ok, {crashed} crashed",
          flush=True)
    return done, crashed


def _link_vicreg_t1(args: argparse.Namespace) -> None:
    """Expose the frozen P0-a2 vicreg/t1 encoders under the W0 root so the
    ladder certificate loop sees a uniform layout (copy, never symlink into
    a frozen root; gates.json comes along for the aggregate tables)."""
    for seed in SEEDS:
        source = Path(args.vicreg_t1_root) / "t1" / "vicreg" / f"s{seed}"
        target = Path(args.output) / "t1" / "vicreg" / f"s{seed}"
        if (target / "encoder.pt").exists():
            continue
        if not (source / "encoder.pt").exists():
            print(f"[v20-w0] WARNING: missing frozen reference {source}",
                  flush=True)
            continue
        target.mkdir(parents=True, exist_ok=True)
        for name in ("encoder.pt", "gates.json"):
            if (source / name).exists():
                shutil.copy2(source / name, target / name)
        print(f"[v20-w0] linked frozen vicreg/t1/s{seed} reference", flush=True)


def _run_certificates(arms: list[str], args: argparse.Namespace,
                      log_dir: Path) -> None:
    """One certify process per task, parallel over GPUs (banks are shared
    within a process across arms/seeds only via regeneration — the per-task
    split keeps MuJoCo renders from racing)."""
    running: list[tuple[subprocess.Popen, str]] = []
    for index, task in enumerate(LADDER_ALL):
        gpu = GPUS[index % len(GPUS)]
        command = [sys.executable, str(REPO / "scripts" / "certify_v20_w0.py"),
                   "--root", args.output, "--tasks", task,
                   "--arms", ",".join(arms)]
        log = open(log_dir / f"cert_{task}.log", "w")
        running.append((subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT,
            env=_job_env(gpu, wandb_on=False), cwd=REPO), task))
        print(f"[v20-w0] certify {task} -> gpu{gpu}", flush=True)
    failures = 0
    for proc, task in running:
        code = proc.wait()
        failures += code != 0
        print(f"[v20-w0] certify {task}: exit {code}", flush=True)
    if failures:
        print(f"[v20-w0] WARNING: {failures} certificate job(s) failed — "
              f"see {log_dir}/cert_*.log", flush=True)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    log_dir = Path(args.output) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    _pregenerate_caches(args, log_dir)

    wave1 = ([Job("t1", arm, seed) for arm in VISREG_ARMS for seed in SEEDS]
             + [Job(task, "vicreg", seed)
                for task in LADDER_NEW for seed in SEEDS])
    _run_wave(wave1, args, log_dir, "wave-1")

    lambda_star = w0_aggregate.select_lambda_star(Path(args.output))
    print(f"[v20-w0] lambda* = {lambda_star}", flush=True)

    certificate_arms = ["vicreg"]
    if lambda_star is None:
        print("[v20-w0] claim 1 FAIL-CLOSED: no visreg lambda passed all t1 "
              "seeds; skipping wave-2 (vicreg fallback stands for W1)",
              flush=True)
    else:
        wave2 = ([Job(task, lambda_star, seed)
                  for task in ("t3", "t4") for seed in SEEDS]
                 + [Job(task, lambda_star, seed)
                    for task in LADDER_NEW for seed in SEEDS])
        _run_wave(wave2, args, log_dir, "wave-2")
        certificate_arms.append(lambda_star)

    _link_vicreg_t1(args)
    _run_certificates(certificate_arms, args, log_dir)

    subprocess.run([sys.executable,
                    str(REPO / "scripts" / "aggregate_v20_w0.py"),
                    "--root", args.output], cwd=REPO, check=False)


if __name__ == "__main__":
    main()
