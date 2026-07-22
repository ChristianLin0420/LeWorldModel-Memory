#!/usr/bin/env python3
"""Three-GPU launcher for the raw OGBench CEM campaign."""
from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/bin/python"
RUNNER = ROOT / "scripts/run_cem_raw_ogbench.py"
DEFAULT_OUTPUT = ROOT / "outputs/cem_raw_ogbench"
SMOKE_OUTPUT = ROOT / "outputs/cem_raw_ogbench_smoke"

FOCUSED = (
    "pointmaze-large-navigate-v0",
    "pointmaze-teleport-navigate-v0",
    "cube-single-play-v0",
    "cube-triple-play-v0",
)
BREADTH = (
    "puzzle-3x3-play-v0",
    "scene-play-v0",
    "pointmaze-giant-navigate-v0",
    "cube-double-play-v0",
    "antmaze-large-navigate-v0",
    "humanoidmaze-large-navigate-v0",
)
SMOKE_ENVS = (
    "pointmaze-large-navigate-v0",
    "cube-single-play-v0",
)


def stable_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def command(
    output: Path,
    env_name: str,
    gpu: int,
    *,
    seed: int | None = None,
    prepare: bool = False,
    smoke: bool = False,
    overwrite: bool = False,
) -> list[str]:
    args = [
        str(PYTHON),
        str(RUNNER),
        "--output",
        str(output),
        "--env-name",
        env_name,
        "--gpu",
        str(gpu),
    ]
    if prepare:
        args.append("--prepare-features")
    if seed is not None:
        args.extend(["--seed", str(seed)])
    if smoke:
        args.append("--smoke")
    if overwrite:
        args.append("--overwrite")
    return args


def run_pool(
    jobs: list[dict[str, Any]],
    output: Path,
    gpus: list[int],
) -> list[dict[str, Any]]:
    logs = output / "launch_logs"
    logs.mkdir(parents=True, exist_ok=True)
    pending = deque(jobs)
    available = deque(gpus)
    active: list[dict[str, Any]] = []
    finished = []
    while pending or active:
        while pending and available:
            job = pending.popleft()
            gpu = available.popleft()
            name = job["name"]
            log_path = logs / f"{name}.log"
            stream = log_path.open("w")
            argv = command(
                output,
                job["env"],
                gpu,
                seed=job.get("seed"),
                prepare=job.get("prepare", False),
                smoke=job.get("smoke", False),
                overwrite=job.get("overwrite", False),
            )
            process = subprocess.Popen(
                argv,
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active.append({
                **job,
                "gpu": gpu,
                "pid": process.pid,
                "process": process,
                "stream": stream,
                "log": str(log_path.relative_to(ROOT)),
                "started_at": time.time(),
                "argv": argv,
            })
            print(
                f"[launch] gpu={gpu} pid={process.pid} {name}",
                flush=True,
            )
        time.sleep(1.0)
        still_active = []
        for job in active:
            return_code = job["process"].poll()
            if return_code is None:
                still_active.append(job)
                continue
            job["stream"].close()
            available.append(job["gpu"])
            record = {
                key: value for key, value in job.items()
                if key not in {"process", "stream"}
            }
            record["return_code"] = int(return_code)
            record["elapsed_seconds"] = (
                time.time() - job["started_at"]
            )
            finished.append(record)
            print(
                f"[done] rc={return_code} gpu={job['gpu']} "
                f"{job['name']}",
                flush=True,
            )
        active = still_active
    return finished


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--campaign", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--envs", nargs="*")
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-cells", action="store_true")
    parser.add_argument(
        "--retry-failures", type=int, default=2,
        help="Retries each failed subprocess this many times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.gpus or any(gpu not in (0, 1, 2) for gpu in args.gpus):
        raise ValueError("allowed physical GPUs are exactly 0, 1, and 2")
    output = args.output or (
        SMOKE_OUTPUT if args.smoke else DEFAULT_OUTPUT
    )
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        envs = tuple(args.envs or SMOKE_ENVS)
        matrix = [(env_name, 0) for env_name in envs]
    else:
        envs = tuple(args.envs or (FOCUSED + BREADTH))
        seeds = tuple(args.seeds or (0, 1, 2))
        matrix = []
        for env_name in envs:
            cell_seeds = seeds if env_name in FOCUSED else (seeds[0],)
            matrix.extend((env_name, seed) for seed in cell_seeds)
    prepare_jobs = [{
        "name": f"prepare_{env_name}",
        "env": env_name,
        "prepare": True,
        "smoke": bool(args.smoke),
        "overwrite": bool(args.overwrite),
    } for env_name in envs]
    prepare_records = run_pool(prepare_jobs, output, args.gpus)
    failures = [
        record for record in prepare_records
        if record["return_code"] != 0
    ]
    for attempt in range(args.retry_failures):
        if not failures:
            break
        print(
            f"[retry] feature preparation attempt {attempt + 1}: "
            f"{len(failures)} jobs",
            flush=True,
        )
        retry_jobs = [{
            "name": f"{record['name']}_retry{attempt + 1}",
            "env": record["env"],
            "prepare": True,
            "smoke": bool(args.smoke),
            "overwrite": False,
        } for record in failures]
        retry_records = run_pool(retry_jobs, output, args.gpus)
        prepare_records.extend(retry_records)
        failures = [
            record for record in retry_records
            if record["return_code"] != 0
        ]
    prepared_envs = {
        record["env"] for record in prepare_records
        if record["return_code"] == 0
    }
    cell_jobs = [{
        "name": f"cell_{env_name}_s{seed}",
        "env": env_name,
        "seed": seed,
        "smoke": bool(args.smoke),
        "overwrite": bool(args.overwrite or args.overwrite_cells),
    } for env_name, seed in matrix if env_name in prepared_envs]
    cell_records = run_pool(cell_jobs, output, args.gpus)
    failures = [
        record for record in cell_records if record["return_code"] != 0
    ]
    for attempt in range(args.retry_failures):
        if not failures:
            break
        print(
            f"[retry] cell attempt {attempt + 1}: {len(failures)} jobs",
            flush=True,
        )
        retry_jobs = [{
            "name": f"{record['name']}_retry{attempt + 1}",
            "env": record["env"],
            "seed": record["seed"],
            "smoke": bool(args.smoke),
            "overwrite": False,
        } for record in failures]
        retry_records = run_pool(retry_jobs, output, args.gpus)
        cell_records.extend(retry_records)
        failures = [
            record for record in retry_records
            if record["return_code"] != 0
        ]
    aggregate_command = [
        str(PYTHON),
        str(RUNNER),
        "--output",
        str(output),
        "--aggregate",
    ]
    aggregate_log = output / "launch_logs/aggregate.log"
    with aggregate_log.open("w") as stream:
        aggregate_result = subprocess.run(
            aggregate_command,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    receipt = {
        "schema": "cem_raw_ogbench_launch_receipt",
        "mode": "smoke" if args.smoke else "campaign",
        "allowed_gpus": args.gpus,
        "gpu3_used": False,
        "environments_requested": list(envs),
        "cell_matrix": [
            {"environment": env_name, "seed": seed}
            for env_name, seed in matrix
        ],
        "prepare_jobs": prepare_records,
        "cell_jobs": cell_records,
        "unrecovered_failures": failures,
        "aggregate_return_code": int(aggregate_result.returncode),
        "aggregate_log": str(aggregate_log.relative_to(ROOT)),
        "jobs_still_running": [],
    }
    (output / "launch_receipt.json").write_text(stable_json(receipt))
    if failures or aggregate_result.returncode != 0:
        raise RuntimeError(
            f"campaign completed with {len(failures)} failed cells; "
            f"see {output / 'launch_receipt.json'}"
        )
    print(stable_json({
        "status": "completed",
        "output": str(output),
        "cell_count": len(matrix),
        "receipt": str(output / "launch_receipt.json"),
    }))


if __name__ == "__main__":
    main()
