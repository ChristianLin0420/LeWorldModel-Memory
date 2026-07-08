#!/usr/bin/env python3
"""Preview or run adaptive Wave-1b on physical GPU 0 only."""

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

from scripts.paper_a_matched_color_spec import (  # noqa: E402
    ARMS, DEFAULT_SHA, DEFAULT_SPEC, HOSTS, SEEDS, load_locked_spec,
    output_path,
)
from scripts.prepare_paper_a_matched_color import host_manifest_path  # noqa: E402
from scripts.train_paper_a_matched_color import carrier_directory  # noqa: E402


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("prepare", "carriers", "aggregate", "all"),
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
        raise RuntimeError(f"Wave-1b requires physical GPU 0; got {visible!r}")
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


def _base(script: str, spec: Path, sha: Path,
          *arguments: str) -> list[str]:
    return [str(ROOT / ".venv/bin/python"), script, *arguments,
            "--spec", str(spec.resolve()), "--sha", str(sha.resolve()),
            "--execute"]


def _carrier_commands(spec: Path, sha: Path
                      ) -> list[tuple[str, str, int, list[str], Path]]:
    result = []
    for host in HOSTS:
        for arm in ARMS:
            for seed in SEEDS:
                command = _base(
                    "scripts/train_paper_a_matched_color.py", spec, sha,
                    "--host", host, "--arm", arm, "--seed", str(seed),
                    "--device", "cuda:0")
                log = (ROOT / "outputs/paper_a_matched_color_v1/logs/carriers"
                       / f"{host}-{arm}-s{seed}.log")
                result.append((host, arm, seed, command, log))
    return result


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if not 1 <= args.workers <= 6:
        raise ValueError("--workers must be in [1,6] for one physical GPU")
    spec = load_locked_spec(args.spec, args.sha, verify_inputs=False)
    prepare = [
        (host, _base(
            "scripts/prepare_paper_a_matched_color.py", args.spec, args.sha,
            "--host", host, "--device", "cuda:0"),
         ROOT / "outputs/paper_a_matched_color_v1/logs/prepare"
         / f"{host}.log")
        for host in HOSTS
    ]
    carriers = _carrier_commands(args.spec, args.sha)
    aggregate = _base(
        "scripts/aggregate_paper_a_matched_color.py", args.spec, args.sha)
    if not args.execute:
        print("Wave-1b preview: 3 preparations, 75 carrier cells, "
              "1 aggregation; physical GPUs={0}")
        if args.phase in ("prepare", "all"):
            for host, command, log in prepare:
                print("prepare", host, " ".join(command), "->", log)
        if args.phase in ("carriers", "all"):
            for host, arm, seed, command, log in carriers:
                print("carrier", host, arm, seed, " ".join(command), "->", log)
        if args.phase in ("aggregate", "all"):
            print("aggregate", " ".join(aggregate))
        return

    if args.phase in ("prepare", "all"):
        for host, command, log in prepare:
            if host_manifest_path(spec, host).is_file():
                if args.resume:
                    continue
                raise FileExistsError(f"Wave-1b prepare exists: {host}")
            _run(command, log)
    if args.phase in ("carriers", "all"):
        for host in HOSTS:
            path = host_manifest_path(spec, host)
            if not path.is_file():
                raise FileNotFoundError(f"Wave-1b admission missing: {host}")
        pending = []
        for host, arm, seed, command, log in carriers:
            complete = (carrier_directory(spec, host, arm, seed)
                        / "manifest.json").is_file()
            if complete and args.resume:
                continue
            if complete:
                raise FileExistsError(
                    f"Wave-1b carrier exists: {host}/{arm}/{seed}")
            pending.append((host, arm, seed, command, log))
        failures: list[tuple[tuple[str, str, int], str]] = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            active: dict[Future[None], tuple[str, str, int]] = {}
            cursor = 0
            while cursor < len(pending) or active:
                while (cursor < len(pending) and len(active) < args.workers
                       and not failures):
                    host, arm, seed, command, log = pending[cursor]
                    active[pool.submit(_run, command, log)] = (host, arm, seed)
                    cursor += 1
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    cell = active.pop(future)
                    try:
                        future.result()
                        print(f"[matched-color/queue] complete {cell}", flush=True)
                    except BaseException as error:
                        failures.append((cell, str(error)))
                        print(f"[matched-color/queue] FAILED {cell}: {error}",
                              flush=True)
                if failures:
                    cursor = len(pending)
        if failures:
            raise RuntimeError(f"Wave-1b carrier failures: {failures}")
    if args.phase in ("aggregate", "all"):
        root = output_path(spec, "root")
        complete = (root / "summary.json").is_file() \
            and (root / "final_audit.json").is_file()
        if complete and args.resume:
            return
        if complete:
            raise FileExistsError("Wave-1b aggregation already exists")
        _run(aggregate, root / "logs/aggregate.log")


if __name__ == "__main__":
    main()
