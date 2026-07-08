#!/usr/bin/env python3
"""Preview or execute the locked Wave-1 queue on physical GPU 0 only."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections.abc import Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.paper_a_matched_host_spec import (  # noqa: E402
    ARMS,
    DEFAULT_SHA,
    DEFAULT_SPEC,
    HOSTS,
    SEEDS,
    load_locked_spec,
    output_path,
)
from scripts.prepare_paper_a_matched_host import host_manifest_path  # noqa: E402
from scripts.train_paper_a_matched_host import carrier_directory  # noqa: E402


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "carriers", "all"),
                        default="all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--sha", type=Path, default=DEFAULT_SHA)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _env() -> dict[str, str]:
    value = dict(os.environ)
    visible = value.get("CUDA_VISIBLE_DEVICES")
    if visible not in (None, "", "0"):
        raise RuntimeError(
            f"Wave-1 initial queue requires physical GPU 0 only; got {visible!r}")
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
        selected = log.with_name(f"{log.stem}.retry-{attempt}{log.suffix}")
    with selected.open("x") as stream:
        stream.write("physical_gpu\t0\nlogical_device\tcuda:0\n")
        stream.write("command\t" + " ".join(command) + "\n")
        stream.flush()
        result = subprocess.run(
            command, cwd=ROOT, env=_env(), stdout=stream,
            stderr=subprocess.STDOUT, text=True, check=False)
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}); see {selected}")


def _prepare_commands(spec_path: Path, sha_path: Path
                      ) -> list[tuple[list[str], Path]]:
    return [
        ([str(ROOT / ".venv/bin/python"),
          "scripts/prepare_paper_a_matched_host.py",
          "--host", host, "--device", "cuda:0",
          "--spec", str(spec_path), "--sha", str(sha_path), "--execute"],
         ROOT / "outputs/paper_a_matched_host_v1/logs/prepare"
         / f"{host}.log")
        for host in HOSTS
    ]


def _carrier_commands(spec_path: Path, sha_path: Path
                      ) -> list[tuple[str, str, int, list[str], Path]]:
    result = []
    for host in HOSTS:
        for arm in ARMS:
            for seed in SEEDS:
                command = [
                    str(ROOT / ".venv/bin/python"),
                    "scripts/train_paper_a_matched_host.py",
                    "--host", host, "--arm", arm, "--seed", str(seed),
                    "--device", "cuda:0", "--spec", str(spec_path),
                    "--sha", str(sha_path), "--execute",
                ]
                log = (ROOT / "outputs/paper_a_matched_host_v1/logs/carriers"
                       / f"{host}-{arm}-s{seed}.log")
                result.append((host, arm, seed, command, log))
    return result


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.workers < 1 or args.workers > 6:
        raise ValueError("--workers must be in [1,6] for one physical GPU")
    spec = load_locked_spec(args.spec, args.sha, verify_inputs=False)
    prepare = _prepare_commands(args.spec.resolve(), args.sha.resolve())
    carriers = _carrier_commands(args.spec.resolve(), args.sha.resolve())
    selected = []
    if args.phase in ("prepare", "all"):
        selected.extend(("prepare", command, log)
                        for command, log in prepare)
    if args.phase in ("carriers", "all"):
        selected.extend(("carrier", command, log)
                        for _, _, _, command, log in carriers)
    if not args.execute:
        print(f"Wave-1 preview: {len(prepare)} preparations, "
              f"{len(carriers)} carrier cells, physical GPUs={{0}}")
        for kind, command, log in selected:
            print(kind, " ".join(command), "->", log.relative_to(ROOT))
        return

    if args.phase in ("prepare", "all"):
        for host, (command, log) in zip(HOSTS, prepare, strict=True):
            if host_manifest_path(spec, host).is_file():
                if args.resume:
                    continue
                raise FileExistsError(f"prepare output already exists for {host}")
            _run(command, log)
    if args.phase not in ("carriers", "all"):
        return
    for host in HOSTS:
        if not host_manifest_path(spec, host).is_file():
            raise FileNotFoundError(f"host preparation missing: {host}")

    pending = []
    for host, arm, seed, command, log in carriers:
        complete = (carrier_directory(spec, host, arm, seed)
                    / "manifest.json").is_file()
        if complete and args.resume:
            continue
        if complete:
            raise FileExistsError(f"carrier output already exists: {host}/{arm}/{seed}")
        pending.append((host, arm, seed, command, log))
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        active: dict[Future[None], tuple[str, str, int]] = {}
        cursor = 0
        while cursor < len(pending) or active:
            while cursor < len(pending) and len(active) < args.workers:
                host, arm, seed, command, log = pending[cursor]
                active[pool.submit(_run, command, log)] = (host, arm, seed)
                cursor += 1
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                cell = active.pop(future)
                try:
                    future.result()
                    print(f"[matched-host/queue] complete {cell}", flush=True)
                except BaseException as error:
                    failures.append((cell, str(error)))
                    print(f"[matched-host/queue] FAILED {cell}: {error}", flush=True)
            if failures:
                # Running subprocesses finish and retain their logs; no new cell
                # is submitted after the first observed failure.
                cursor = len(pending)
            time.sleep(0.05)
    if failures:
        raise RuntimeError(f"Wave-1 carrier failures: {failures}")
    print("[matched-host/queue] all requested carrier cells complete", flush=True)


if __name__ == "__main__":
    main()
