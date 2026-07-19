#!/usr/bin/env python3
"""Launch Stage-D robustness grid for label-free Mem-JEPA host exposure."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/mem_jepa_stage_d"
AGES = (4, 8, 15)
FULL_SEEDS = (0, 1, 2, 3, 4)
ABLATION_VARIANTS = ("no_host", "no_context", "shuffle_targets",
                     "batch_negatives")
ABLATION_SEEDS = (0, 1, 2, 3, 4)


def python_bin() -> str:
    candidate = ROOT / ".venv/bin/python"
    return str(candidate if candidate.exists() else Path(sys.executable))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--slots", type=int, default=6)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--seed-base", type=int, default=9800)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def cell_id(variant: str, seed: int, age: int) -> str:
    return f"{variant}_seed{seed}_age{age}"


def build_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for seed in FULL_SEEDS:
        for age in AGES:
            jobs.append({"variant": "full", "seed": seed, "age": age})
    for variant in ABLATION_VARIANTS:
        for seed in ABLATION_SEEDS:
            jobs.append({"variant": variant, "seed": seed, "age": 15})
    for index, job in enumerate(jobs):
        variant_offset = ["full", *ABLATION_VARIANTS].index(job["variant"]) * 1000
        job["run_seed"] = int(args.seed_base + variant_offset
                              + 100 * int(job["seed"]) + int(job["age"]))
    return jobs


def command_for(args: argparse.Namespace, job: dict[str, Any]) -> list[str]:
    output = Path(args.output) / "cells" / cell_id(
        job["variant"], int(job["seed"]), int(job["age"]))
    return [
        python_bin(),
        str(ROOT / "scripts/run_mem_jepa_stage_c.py"),
        "--age", str(job["age"]),
        "--output", str(output),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--eval-batch-size", str(args.eval_batch_size),
        "--dim", str(args.dim),
        "--slots", str(args.slots),
        "--heads", str(args.heads),
        "--temperature", str(args.temperature),
        "--seed", str(job["run_seed"]),
        "--variant", str(job["variant"]),
        "--device", "cuda:0",
    ]


def read_cell(root: Path, variant: str, seed: int, age: int) -> dict[str, Any]:
    path = root / "cells" / cell_id(variant, seed, age) / f"age_{age}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def summarize_values(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "stdev": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
    }


def aggregate(output: Path) -> dict[str, Any]:
    full: dict[str, Any] = {}
    all_passed = True
    for age in AGES:
        cells = [read_cell(output, "full", seed, age) for seed in FULL_SEEDS]
        full_metrics = [
            cell["metrics"]["records"]["full"]["balanced_accuracy"]
            for cell in cells]
        reset_metrics = [
            cell["metrics"]["records"]["reset"]["balanced_accuracy"]
            for cell in cells]
        nostate_metrics = [
            cell["metrics"]["records"]["no_state"]["balanced_accuracy"]
            for cell in cells]
        retrieval_metrics = [
            cell["metrics"]["records"]["full"]["candidate_retrieval_accuracy"]
            for cell in cells]
        passes = [bool(cell["metrics"]["gate"]["passed"]) for cell in cells]
        full[str(age)] = {
            "seeds": list(FULL_SEEDS),
            "pass_count": int(sum(passes)),
            "cell_count": len(cells),
            "all_passed": bool(all(passes)),
            "full": summarize_values(full_metrics),
            "reset": summarize_values(reset_metrics),
            "no_state": summarize_values(nostate_metrics),
            "candidate_retrieval": summarize_values(retrieval_metrics),
        }
        all_passed = all_passed and all(passes)

    ablations: dict[str, Any] = {}
    for variant in ABLATION_VARIANTS:
        cells = [read_cell(output, variant, seed, 15)
                 for seed in ABLATION_SEEDS]
        full_metrics = [
            cell["metrics"]["records"]["full"]["balanced_accuracy"]
            for cell in cells]
        reset_metrics = [
            cell["metrics"]["records"]["reset"]["balanced_accuracy"]
            for cell in cells]
        nostate_metrics = [
            cell["metrics"]["records"]["no_state"]["balanced_accuracy"]
            for cell in cells]
        retrieval_metrics = [
            cell["metrics"]["records"]["full"]["candidate_retrieval_accuracy"]
            for cell in cells]
        passes = [bool(cell["metrics"]["gate"]["passed"]) for cell in cells]
        ablations[variant] = {
            "age": 15,
            "seeds": list(ABLATION_SEEDS),
            "pass_count": int(sum(passes)),
            "cell_count": len(cells),
            "all_passed": bool(all(passes)),
            "full": summarize_values(full_metrics),
            "reset": summarize_values(reset_metrics),
            "no_state": summarize_values(nostate_metrics),
            "candidate_retrieval": summarize_values(retrieval_metrics),
        }

    summary = {
        "schema": "mem_jepa_stage_d_summary_v1",
        "status": "completed" if all_passed else "completed_with_full_failures",
        "full_objective_all_seeds_passed": bool(all_passed),
        "labels_used_for_adapter_training": False,
        "full": full,
        "ablations_age15": ablations,
        "updated_unix": time.time(),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    log_dir = output / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args)
    if args.dry_run:
        for job in jobs:
            print(cell_id(job["variant"], job["seed"], job["age"]))
        return

    queue = list(jobs)
    running: dict[int, tuple[dict[str, Any], subprocess.Popen[bytes], object]] = {}
    failed: list[tuple[dict[str, Any], int]] = []
    gpus = [0, 1, 2]
    while queue or running:
        for gpu in gpus:
            if gpu in running or not queue:
                continue
            job = queue.pop(0)
            log_path = log_dir / f"{cell_id(job['variant'], job['seed'], job['age'])}.log"
            log = log_path.open("wb")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            process = subprocess.Popen(
                command_for(args, job), cwd=ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT)
            running[gpu] = (job, process, log)
            print(
                f"[stage-d-launch] gpu={gpu} pid={process.pid} "
                f"cell={cell_id(job['variant'], job['seed'], job['age'])}",
                flush=True,
            )
        time.sleep(10)
        for gpu, (job, process, log) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            print(
                f"[stage-d-done] gpu={gpu} code={code} "
                f"cell={cell_id(job['variant'], job['seed'], job['age'])}",
                flush=True,
            )
            if code != 0:
                failed.append((job, int(code)))
            del running[gpu]
        if failed:
            for _, (_, process, log) in list(running.items()):
                if process.poll() is None:
                    process.terminate()
                log.close()
            raise SystemExit(f"Stage-D jobs failed: {failed}")
    summary = aggregate(output)
    print(json.dumps({
        "status": summary["status"],
        "full_objective_all_seeds_passed":
            summary["full_objective_all_seeds_passed"],
        "full_ages": {
            age: value["pass_count"] for age, value in summary["full"].items()
        },
        "ablations_age15": {
            name: value["pass_count"]
            for name, value in summary["ablations_age15"].items()
        },
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
