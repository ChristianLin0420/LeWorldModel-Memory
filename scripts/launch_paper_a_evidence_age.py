#!/usr/bin/env python3
"""Preview or run the immutable evidence-age waves on physical GPU 0."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.paper_a_evidence_age_spec import (
    ARMS,
    DEFAULT_SHA,
    DEFAULT_SPEC,
    HOSTS,
    SEEDS,
    host_tasks,
    load_locked_spec,
    output_root,
)
from scripts.train_paper_a_evidence_age_strict import carrier_directory


PYTHON = sys.executable
WAVES = ("read-time", "strict-prepare", "strict-carriers",
         "aggregate-read-time", "aggregate-strict")


@dataclass(frozen=True)
class Job:
    name: str
    command: tuple[str, ...]
    done: tuple[Path, ...]


def _common(spec_path: Path, sha_path: Path) -> tuple[str, ...]:
    return ("--spec", str(spec_path), "--sha", str(sha_path), "--execute")


def build_jobs(spec: dict, wave: str, spec_path: Path,
               sha_path: Path) -> list[Job]:
    common = _common(spec_path, sha_path)
    jobs = []
    if wave == "read-time":
        for host in HOSTS:
            for task in host_tasks(spec, host):
                for seed in SEEDS:
                    done = (output_root(spec, "read_time") / host / task
                            / f"seed-{seed}" / "metrics.json")
                    jobs.append(Job(
                        f"read-time-{host}-{task}-s{seed}",
                        (PYTHON, "scripts/run_paper_a_evidence_age_readtime.py",
                         "--host", host, "--task", task, "--seed", str(seed),
                         "--device", "cuda:0", *common), (done,)))
    elif wave == "strict-prepare":
        for host in HOSTS:
            for task in host_tasks(spec, host):
                root = output_root(spec, "strict") / "cache" / host / task
                jobs.append(Job(
                    f"strict-prepare-{host}-{task}",
                    (PYTHON, "scripts/prepare_paper_a_evidence_age_strict.py",
                     "--host", host, "--task", task, "--device", "cuda:0",
                     *common),
                    (root / "manifest.json", root / "prerequisite-stopped.json")))
    elif wave == "strict-carriers":
        for host in HOSTS:
            for task in host_tasks(spec, host):
                cache_manifest = (output_root(spec, "strict") / "cache" / host
                                  / task / "manifest.json")
                if not cache_manifest.is_file() \
                        or json.loads(cache_manifest.read_text()).get("status") \
                        != "admitted":
                    raise RuntimeError(
                        f"strict cache is not admitted for {host}/{task}")
                for arm in ARMS:
                    for seed in SEEDS:
                        directory = carrier_directory(
                            spec, host, task, arm, seed)
                        jobs.append(Job(
                            f"strict-carrier-{host}-{task}-{arm}-s{seed}",
                            (PYTHON, "scripts/train_paper_a_evidence_age_strict.py",
                             "--host", host, "--task", task, "--arm", arm,
                             "--seed", str(seed), "--device", "cuda:0", *common),
                            (directory / "manifest.json",)))
    elif wave == "aggregate-read-time":
        jobs.append(Job(
            wave,
            (PYTHON, "scripts/aggregate_paper_a_evidence_age_readtime.py",
             *common),
            (output_root(spec, "read_time") / "summary.json",)))
    elif wave == "aggregate-strict":
        jobs.append(Job(
            wave,
            (PYTHON, "scripts/aggregate_paper_a_evidence_age_strict.py",
             *common),
            (output_root(spec, "strict") / "summary.json",)))
    else:
        raise ValueError(f"unknown wave {wave!r}")
    return jobs


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave", required=True, choices=(*WAVES, "all"))
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--sha", type=Path, default=DEFAULT_SHA)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def _complete(job: Job) -> bool:
    return any(path.is_file() for path in job.done)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    spec = load_locked_spec(args.spec, args.sha)
    waves = WAVES if args.wave == "all" else (args.wave,)
    plan = [(wave, build_jobs(
        spec, wave, args.spec.resolve(), args.sha.resolve())) for wave in waves]
    for wave, jobs in plan:
        for job in jobs:
            print(f"{wave}\t{'complete' if _complete(job) else 'pending'}\t"
                  f"{job.name}\t{shlex.join(job.command)}")
    if not args.execute:
        return
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    environment.setdefault("PYTHONHASHSEED", "0")
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                     "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment.setdefault(variable, "1")
    logs = output_root(spec, "logs")
    logs.mkdir(parents=True, exist_ok=True)
    for wave, jobs in plan:
        for job in jobs:
            if _complete(job):
                continue
            log = logs / f"{job.name}.log"
            if log.exists():
                raise FileExistsError(
                    f"pending job has an existing immutable log: {log}")
            with log.open("x") as stream:
                process = subprocess.run(
                    job.command, cwd=ROOT, env=environment,
                    stdout=stream, stderr=subprocess.STDOUT, check=False)
            if process.returncode != 0 or not _complete(job):
                raise SystemExit(
                    f"job {job.name} failed with {process.returncode}; see {log}")
            print(f"[evidence-age-launch] done {job.name}", flush=True)


if __name__ == "__main__":
    main()
