#!/usr/bin/env python3
"""Run matched-token admissions, carriers, aggregation, and use on GPU0."""

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

from scripts.evaluate_paper_a_matched_token_use import use_cell_directory  # noqa: E402
from scripts.paper_a_matched_token_spec import (  # noqa: E402
    ARMS, DEFAULT_SHA, DEFAULT_SPEC, HOSTS, SEEDS, load_locked_spec,
    output_path,
)
from scripts.prepare_paper_a_matched_token import host_manifest_path  # noqa: E402
from scripts.prepare_paper_a_matched_token_use import deck_path, gate_path  # noqa: E402
from scripts.train_paper_a_matched_token import carrier_directory  # noqa: E402


def parse_args(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=(
        "prepare", "carriers", "aggregate", "use-prepare", "use-cells",
        "use-aggregate", "use", "all"), default="all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--sha", type=Path, default=DEFAULT_SHA)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _env():
    value = dict(os.environ)
    if value.get("CUDA_VISIBLE_DEVICES") not in (None, "", "0"):
        raise RuntimeError("matched-token queue requires physical GPU0")
    value.update({"CUDA_VISIBLE_DEVICES": "0",
                  "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "0"})
    return value


def _run(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    selected, attempt = log, 0
    while selected.exists():
        attempt += 1
        selected = log.with_name(f"{log.stem}.retry-{attempt}{log.suffix}")
    with selected.open("x") as stream:
        stream.write("physical_gpu\t0\ncommand\t" + " ".join(command) + "\n")
        stream.flush()
        result = subprocess.run(command, cwd=ROOT, env=_env(), stdout=stream,
                                stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise RuntimeError(f"command failed; see {selected}")


def _cmd(script, spec, sha, *extra):
    return [str(ROOT / ".venv/bin/python"), script, *extra,
            "--spec", str(spec.resolve()), "--sha", str(sha.resolve()),
            "--execute"]


def carrier_commands(spec, sha):
    return [(host, arm, seed, _cmd(
        "scripts/train_paper_a_matched_token.py", spec, sha,
        "--host", host, "--arm", arm, "--seed", str(seed),
        "--device", "cuda:0")) for host in HOSTS for arm in ARMS for seed in SEEDS]


def use_commands(spec, sha):
    return [(arm, seed, _cmd(
        "scripts/evaluate_paper_a_matched_token_use.py", spec, sha,
        "--arm", arm, "--seed", str(seed), "--device", "cuda:0"))
        for arm in ARMS for seed in SEEDS]


def _parallel(jobs, workers, root, directory, complete_fn, resume):
    pending = []
    for identity, command in jobs:
        if complete_fn(identity):
            if resume:
                continue
            raise FileExistsError(f"completed cell exists: {identity}")
        pending.append((identity, command))
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        active: dict[Future[None], tuple] = {}
        cursor = 0
        while cursor < len(pending) or active:
            while cursor < len(pending) and len(active) < workers and not failures:
                identity, command = pending[cursor]
                name = "-".join(map(str, identity))
                active[pool.submit(_run, command,
                                   root / directory / f"{name}.log")] = identity
                cursor += 1
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                identity = active.pop(future)
                try:
                    future.result()
                    print(f"[matched-token/queue] complete {identity}", flush=True)
                except BaseException as error:
                    failures.append((identity, str(error)))
            if failures:
                cursor = len(pending)
    if failures:
        raise RuntimeError(f"matched-token failures: {failures}")


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if not 1 <= args.workers <= 6:
        raise ValueError("workers must be in [1,6]")
    spec = load_locked_spec(args.spec, args.sha, verify_inputs=False)
    root, use_root = output_path(spec, "root"), output_path(spec, "use")
    carriers, uses = carrier_commands(args.spec, args.sha), use_commands(
        args.spec, args.sha)
    if not args.execute:
        print(f"matched-token preview: 3 prep, {len(carriers)} carriers, "
              f"{len(uses)} use cells, GPU={{0}}")
        return
    if args.phase in ("prepare", "all"):
        for host in HOSTS:
            path = host_manifest_path(spec, host)
            if path.exists() and args.resume:
                continue
            if path.exists():
                raise FileExistsError(path)
            _run(_cmd("scripts/prepare_paper_a_matched_token.py",
                      args.spec, args.sha, "--host", host,
                      "--device", "cuda:0"), root / "logs/prepare" / f"{host}.log")
    if args.phase in ("carriers", "all"):
        jobs = [((h, a, s), command) for h, a, s, command in carriers]
        _parallel(jobs, args.workers, root, "logs/carriers",
                  lambda x: (carrier_directory(spec, *x) / "manifest.json").is_file(),
                  args.resume)
    if args.phase in ("aggregate", "all"):
        if not (root / "summary.json").exists():
            _run(_cmd("scripts/aggregate_paper_a_matched_token.py",
                      args.spec, args.sha), root / "logs/aggregate.log")
    if args.phase in ("use-prepare", "use", "all"):
        if not deck_path(spec).exists():
            _run(_cmd("scripts/prepare_paper_a_matched_token_use.py",
                      args.spec, args.sha, "--device", "cuda:0"),
                 use_root / "logs/prepare.log")
    if args.phase in ("use-cells", "use", "all"):
        jobs = [((a, s), command) for a, s, command in uses]
        _parallel(jobs, args.workers, use_root, "logs/cells",
                  lambda x: (use_cell_directory(spec, *x)
                             / "manifest.json").is_file(), args.resume)
    if args.phase in ("use-aggregate", "use", "all"):
        if not (use_root / "summary.json").exists():
            _run(_cmd("scripts/aggregate_paper_a_matched_token_use.py",
                      args.spec, args.sha), use_root / "logs/aggregate.log")


if __name__ == "__main__":
    main()
