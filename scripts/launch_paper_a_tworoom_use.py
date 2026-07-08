#!/usr/bin/env python3
"""Preview or execute the locked 1+25+1 TwoRoom external-use queue."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.aggregate_paper_a_tworoom_use import use_cell_directory  # noqa: E402
from scripts.paper_a_matched_host_spec import (  # noqa: E402
    ARMS,
    DEFAULT_SHA,
    DEFAULT_SPEC,
    SEEDS,
    load_locked_spec,
    output_path,
)
from scripts.prepare_paper_a_tworoom_use import deck_path, gate_path  # noqa: E402


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("prepare", "cells", "aggregate", "all"),
        default="all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--sha", type=Path, default=DEFAULT_SHA)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _environment() -> dict[str, str]:
    value = dict(os.environ)
    visible = value.get("CUDA_VISIBLE_DEVICES")
    if visible not in (None, "", "0"):
        raise RuntimeError(
            f"TwoRoom use queue requires physical GPU 0; got {visible!r}")
    value["CUDA_VISIBLE_DEVICES"] = "0"
    value["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    value["PYTHONHASHSEED"] = "0"
    return value


def _run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    selected = log
    attempt = 0
    while selected.exists():
        attempt += 1
        selected = log.with_name(
            f"{log.stem}.retry-{attempt}{log.suffix}")
    with selected.open("x") as stream:
        stream.write("physical_gpu\t0\nlogical_device\tcuda:0\n")
        stream.write("command\t" + " ".join(command) + "\n")
        stream.flush()
        result = subprocess.run(
            command, cwd=ROOT, env=_environment(), stdout=stream,
            stderr=subprocess.STDOUT, text=True, check=False)
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}); see {selected}")


def _command(script: str, spec: Path, sha: Path,
             *arguments: str) -> list[str]:
    return [
        str(ROOT / ".venv/bin/python"), script, *arguments,
        "--spec", str(spec.resolve()), "--sha", str(sha.resolve()),
        "--execute",
    ]


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if not 1 <= args.workers <= 6:
        raise ValueError("--workers must be in [1,6] for one physical GPU")
    spec = load_locked_spec(args.spec, args.sha, verify_inputs=False)
    use_root = output_path(spec, "use")
    prepare = _command(
        "scripts/prepare_paper_a_tworoom_use.py", args.spec, args.sha,
        "--device", "cuda:0")
    cells = [
        (arm, seed, _command(
            "scripts/evaluate_paper_a_tworoom_use.py", args.spec, args.sha,
            "--arm", arm, "--seed", str(seed), "--device", "cuda:0"))
        for arm in ARMS for seed in SEEDS
    ]
    aggregate = _command(
        "scripts/aggregate_paper_a_tworoom_use.py", args.spec, args.sha)
    if not args.execute:
        print("TwoRoom use preview: 1 deck, 25 cells, 1 aggregation; "
              "physical GPUs={0}")
        if args.phase in ("prepare", "all"):
            print("prepare", " ".join(prepare))
        if args.phase in ("cells", "all"):
            for arm, seed, command in cells:
                print("cell", arm, seed, " ".join(command))
        if args.phase in ("aggregate", "all"):
            print("aggregate", " ".join(aggregate))
        return

    if args.phase in ("prepare", "all"):
        complete = deck_path(spec).is_file() and gate_path(spec).is_file()
        if complete and not args.resume:
            raise FileExistsError("TwoRoom use deck already exists")
        if not complete:
            _run(prepare, use_root / "logs/prepare.log")
    if args.phase in ("cells", "all"):
        if not deck_path(spec).is_file() or not gate_path(spec).is_file():
            raise FileNotFoundError("TwoRoom use deck/gate is missing")
        pending = []
        for arm, seed, command in cells:
            complete = (use_cell_directory(spec, arm, seed)
                        / "manifest.json").is_file()
            if complete and args.resume:
                continue
            if complete:
                raise FileExistsError(f"use cell exists: {arm}/seed-{seed}")
            pending.append((arm, seed, command))
        failures: list[tuple[tuple[str, int], str]] = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            active: dict[Future[None], tuple[str, int]] = {}
            cursor = 0
            while cursor < len(pending) or active:
                while (cursor < len(pending) and len(active) < args.workers
                       and not failures):
                    arm, seed, command = pending[cursor]
                    log = use_root / "logs/cells" / f"{arm}-s{seed}.log"
                    active[pool.submit(_run, command, log)] = (arm, seed)
                    cursor += 1
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    cell = active.pop(future)
                    try:
                        future.result()
                        print(f"[matched-host/use-queue] complete {cell}",
                              flush=True)
                    except BaseException as error:
                        failures.append((cell, str(error)))
                if failures:
                    cursor = len(pending)
        if failures:
            raise RuntimeError(f"TwoRoom use cell failures: {failures}")
    if args.phase in ("aggregate", "all"):
        summary = use_root / "summary.json"
        audit = use_root / "final_audit.json"
        complete = summary.is_file() and audit.is_file()
        if complete and args.resume:
            return
        if complete:
            raise FileExistsError("TwoRoom use aggregation already exists")
        _run(aggregate, use_root / "logs/aggregate.log")


if __name__ == "__main__":
    main()
