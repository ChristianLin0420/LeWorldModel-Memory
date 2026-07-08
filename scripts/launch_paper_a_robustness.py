#!/usr/bin/env python3
"""Plan or explicitly execute the locked Paper-A robustness waves.

The default mode is a read-only command preview.  Actual execution requires
``--execute`` and a device allowed by the locked specification; CUDA devices
0 and 3 are rejected before any output directory is created.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.cache_paper_a_fresh_validation import cache_directory  # noqa: E402
from scripts.evaluate_paper_a_fresh_validation import evaluation_directory  # noqa: E402
from scripts.make_paper_a_robustness_data import bank_path  # noqa: E402
from scripts.paper_a_robustness_spec import (  # noqa: E402
    DEFAULT_SPEC,
    load_locked_spec,
    resolve_spec_path,
    validate_device,
)


PYTHON = sys.executable
WAVES = ("fresh-data", "fresh-cache", "fresh-eval", "seed-extension")


@dataclass(frozen=True)
class Job:
    name: str
    command: tuple[str, ...]
    done_file: Path


def build_jobs(spec: Mapping, wave: str, device: str,
               spec_path: Path = DEFAULT_SPEC) -> list[Job]:
    if wave not in WAVES:
        raise ValueError(f"unknown robustness wave {wave!r}")
    validate_device(spec, device)
    tasks = tuple(spec["tasks"])
    banks = tuple(bank["id"] for bank in spec["fresh_validation"]["banks"])
    jobs: list[Job] = []
    if wave == "fresh-data":
        egl_device = device.rsplit(":", 1)[-1]
        for task in tasks:
            for bank in banks:
                jobs.append(Job(
                    f"{task}_{bank}",
                    (PYTHON, "scripts/make_paper_a_robustness_data.py",
                     "--spec", str(spec_path), "--task", task,
                     "--bank", bank, "--egl-device-id", egl_device),
                    bank_path(spec, task, bank),
                ))
    elif wave == "fresh-cache":
        for task in tasks:
            for bank in banks:
                jobs.append(Job(
                    f"{task}_{bank}",
                    (PYTHON, "scripts/cache_paper_a_fresh_validation.py",
                     "--spec", str(spec_path), "--task", task,
                     "--bank", bank, "--device", device),
                    cache_directory(spec, task, bank) / "manifest.json",
                ))
    elif wave == "fresh-eval":
        fresh = spec["fresh_validation"]
        for task in tasks:
            for bank in banks:
                for arm in fresh["checkpoint_arms"]:
                    for seed in fresh["checkpoint_seeds"]:
                        jobs.append(Job(
                            f"{task}_{bank}_{arm}_s{seed}",
                            (PYTHON,
                             "scripts/evaluate_paper_a_fresh_validation.py",
                             "--spec", str(spec_path), "--task", task,
                             "--bank", bank, "--arm", arm,
                             "--seed", str(seed), "--device", device),
                            evaluation_directory(
                                spec, task, bank, arm, int(seed)) / "metrics.json",
                        ))
    else:
        extension = spec["carrier_seed_extension"]
        output = resolve_spec_path(
            spec, spec["output"]["carrier_seed_extension"])
        cache_root = resolve_spec_path(spec, spec["parent"]["cache_root"])
        weights = resolve_spec_path(
            spec, spec["parent"]["official_weights"]["path"])
        for task in tasks:
            for arm in extension["arms"]:
                for seed in extension["seeds"]:
                    jobs.append(Job(
                        f"{task}_{arm}_s{seed}",
                        (PYTHON, "scripts/train_frozen_official_swap.py",
                         "--task", task, "--arm", arm,
                         "--seed", str(seed), "--cache-root", str(cache_root),
                         "--weights", str(weights), "--output", str(output),
                         "--epochs", str(extension["epochs"]),
                         "--batch-size", str(extension["batch_size"]),
                         "--lr", str(extension["learning_rate"]),
                         "--weight-decay", str(extension["weight_decay"]),
                         "--device", device,
                         "--study",
                         "official-lewm-frozen-carrier-seed-extension-v1",
                         "--provenance-spec", str(spec_path)),
                        output / task / arm / f"s{seed}" / "metrics.json",
                    ))
    names = [job.name for job in jobs]
    destinations = [job.done_file for job in jobs]
    if len(names) != len(set(names)) or len(destinations) != len(set(destinations)):
        raise RuntimeError("robustness job grid contains duplicate cells")
    parent = resolve_spec_path(spec, spec["parent"]["root"])
    for destination in destinations:
        if parent == destination or parent in destination.parents:
            raise RuntimeError(
                f"robustness job would write into parent artifacts: {destination}")
    return jobs


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--wave", required=True, choices=WAVES)
    parser.add_argument("--device")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--execute", action="store_true",
        help="execute commands; without this flag only print the immutable plan")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    spec = load_locked_spec(args.spec)
    device = args.device or spec["execution"]["default_device"]
    validate_device(spec, device)
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard selection")
    canonical = build_jobs(spec, args.wave, device, args.spec.resolve())
    assigned = [job for index, job in enumerate(canonical)
                if index % args.shard_count == args.shard_index]
    pending = [job for job in assigned if not job.done_file.is_file()]
    print(
        f"[robust-launch] wave={args.wave} canonical={len(canonical)} "
        f"assigned={len(assigned)} pending={len(pending)} device={device} "
        f"execute={args.execute}", flush=True)
    if not args.execute:
        for job in pending:
            print(f"{job.name}\t{shlex.join(job.command)}")
        return

    logs_root = resolve_spec_path(spec, spec["output"]["logs"])
    logs = logs_root / args.wave
    logs.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.setdefault("PYTHONHASHSEED", "0")
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                     "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment.setdefault(variable, "1")
    running: list[tuple[subprocess.Popen, Job, object]] = []
    failed = completed = 0
    queue = list(pending)
    while queue or running:
        while queue and len(running) < args.jobs:
            job = queue.pop(0)
            stream = (logs / f"{job.name}.log").open("x")
            process = subprocess.Popen(
                job.command, cwd=ROOT, env=environment,
                stdout=stream, stderr=subprocess.STDOUT)
            running.append((process, job, stream))
            print(f"[robust-launch] start {job.name}", flush=True)
        time.sleep(2)
        active = []
        for process, job, stream in running:
            code = process.poll()
            if code is None:
                active.append((process, job, stream))
                continue
            stream.close()
            if code == 0 and job.done_file.is_file():
                completed += 1
                print(f"[robust-launch] done {job.name}", flush=True)
            else:
                failed += 1
                print(f"[robust-launch] FAIL {job.name} exit={code}", flush=True)
        running = active
    if failed:
        raise SystemExit(f"{failed} robustness jobs failed")
    print(f"[robust-launch] complete={completed}", flush=True)


if __name__ == "__main__":
    main()
