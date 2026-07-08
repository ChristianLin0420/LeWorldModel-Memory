#!/usr/bin/env python3
"""Preview or run adaptive Wave 1.1 on physical GPU 0 only."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path

from lewm.official_tasks.artifacts import atomic_text, stable_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.paper_a_matched_color_v1_1_spec import (  # noqa: E402
    ARMS, DEFAULT_SHA, DEFAULT_SPEC, HOSTS, SEEDS, load_locked_spec,
    output_path, sha256_file,
)
from scripts.prepare_paper_a_matched_color_v1_1 import host_manifest_path  # noqa: E402
from scripts.train_paper_a_matched_color_v1_1 import carrier_directory  # noqa: E402


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
        raise RuntimeError(f"Wave 1.1 requires physical GPU 0; got {visible!r}")
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
                    "scripts/train_paper_a_matched_color_v1_1.py", spec, sha,
                    "--host", host, "--arm", arm, "--seed", str(seed),
                    "--device", "cuda:0")
                log = (ROOT / "outputs/paper_a_matched_color_v1_1/logs/carriers"
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
            "scripts/prepare_paper_a_matched_color_v1_1.py", args.spec, args.sha,
            "--host", host, "--device", "cuda:0"),
         ROOT / "outputs/paper_a_matched_color_v1_1/logs/prepare"
         / f"{host}.log")
        for host in HOSTS
    ]
    carriers = _carrier_commands(args.spec, args.sha)
    aggregate = _base(
        "scripts/aggregate_paper_a_matched_color_v1_1.py", args.spec, args.sha)
    if not args.execute:
        print("Wave 1.1 preview: 2 preparations, 50 carrier cells, "
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

    preparation_failures: list[tuple[str, str]] = []
    if args.phase in ("prepare", "all"):
        for host, command, log in prepare:
            if host_manifest_path(spec, host).is_file():
                if args.resume:
                    continue
                raise FileExistsError(f"Wave 1.1 prepare exists: {host}")
            try:
                _run(command, log)
            except BaseException as error:
                preparation_failures.append((host, str(error)))
                print(f"[matched-color-v1.1/queue] admission resolved nonzero "
                      f"for {host}: {error}", flush=True)

    admissions: dict[str, dict] = {}
    missing: list[str] = []
    for host in HOSTS:
        path = host_manifest_path(spec, host)
        if not path.is_file():
            missing.append(host)
            continue
        admissions[host] = json.loads(path.read_text())
    if missing:
        if preparation_failures:
            raise RuntimeError(
                f"Wave 1.1 prerequisite failures={preparation_failures}; "
                f"missing receipts={missing}")
        if args.phase in ("carriers", "aggregate", "all"):
            raise FileNotFoundError(f"Wave 1.1 admissions missing: {missing}")
    globally_admitted = len(admissions) == len(HOSTS) and all(
        value.get("lock") == spec["_lock"]
        and value.get("status") == "admitted"
        and value.get("all_color_ages_admitted") is True
        and value.get("frozen_host_unchanged") is True
        for value in admissions.values())
    if len(admissions) == len(HOSTS) and not globally_admitted:
        root = output_path(spec, "root")
        stop = root / "global_admission_stop.json"
        payload = {
            "schema_version": 1, "study": spec["study"],
            "lock": spec["_lock"], "status": "stopped-global-admission",
            "no_carrier_training": True,
            "admissions": {
                host: {
                    "path": str(host_manifest_path(spec, host).relative_to(ROOT)),
                    "sha256": sha256_file(host_manifest_path(spec, host)),
                    "status": admissions[host].get("status"),
                    "all_color_ages_admitted": admissions[host].get(
                        "all_color_ages_admitted"),
                }
                for host in HOSTS
            },
            "preparation_process_exit_records": preparation_failures,
        }
        if not stop.exists():
            atomic_text(stop, stable_json(payload))
        if args.phase in ("carriers", "aggregate", "all"):
            print("[matched-color-v1.1/queue] global admission failed; "
                  "no carrier or aggregation launched", flush=True)
            return
    if args.phase in ("carriers", "all"):
        if not globally_admitted:
            raise RuntimeError("Wave 1.1 global admission is not satisfied")
        pending = []
        for host, arm, seed, command, log in carriers:
            complete = (carrier_directory(spec, host, arm, seed)
                        / "manifest.json").is_file()
            if complete and args.resume:
                continue
            if complete:
                raise FileExistsError(
                    f"Wave 1.1 carrier exists: {host}/{arm}/{seed}")
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
                        print(f"[matched-color-v1.1/queue] complete {cell}", flush=True)
                    except BaseException as error:
                        failures.append((cell, str(error)))
                        print(f"[matched-color-v1.1/queue] FAILED {cell}: {error}",
                              flush=True)
                if failures:
                    cursor = len(pending)
        if failures:
            raise RuntimeError(f"Wave 1.1 carrier failures: {failures}")
    if args.phase in ("aggregate", "all"):
        root = output_path(spec, "root")
        complete = (root / "summary.json").is_file() \
            and (root / "final_audit.json").is_file()
        if complete and args.resume:
            return
        if complete:
            raise FileExistsError("Wave 1.1 aggregation already exists")
        _run(aggregate, root / "logs/aggregate.log")


if __name__ == "__main__":
    main()
