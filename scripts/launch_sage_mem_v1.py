#!/usr/bin/env python3
"""Preview or launch SAGE-Mem v1 with strict physical GPU ownership."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_sage_mem_v1 import FORMAL_CONFIRMATION  # noqa: E402
from scripts.sage_mem_v1_spec import (  # noqa: E402
    COHORTS, DEFAULT_SPEC, development_cells, formal_cells, load_spec,
    output_root,
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=(
        "preflight", "development", "smoke", "seal", "prepare", "full"),
        required=True)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--formal-confirmation")
    return parser.parse_args(argv)


def _command(spec: Path, stage: str, *, cohort: str | None = None,
             arm: str | None = None, seed: int | None = None,
             resume: bool = False) -> list[str]:
    command = [str(ROOT / ".venv/bin/python"),
               "scripts/run_sage_mem_v1.py", "--stage", stage,
               "--spec", str(spec.resolve()), "--execute"]
    if cohort is not None:
        command.extend(("--cohort", cohort))
    if arm is not None:
        command.extend(("--arm", arm))
    if seed is not None:
        command.extend(("--seed", str(seed)))
    if stage == "full":
        command.extend(("--formal-confirmation", FORMAL_CONFIRMATION))
    if resume:
        command.append("--resume")
    return command


def planned_commands(spec_path: Path, stage: str,
                     *, resume: bool = False) -> list[tuple[int, list[str]]]:
    spec = load_spec(spec_path)
    if stage in ("preflight", "seal"):
        return [(-1, _command(spec_path, stage, resume=resume))]
    if stage in ("smoke", "prepare"):
        return [(spec["cohorts"][cohort]["gpu"], _command(
            spec_path, stage, cohort=cohort, resume=resume))
                for cohort in COHORTS]
    if stage == "development":
        return [(spec["cohorts"][cohort]["gpu"], _command(
            spec_path, "development", cohort=cohort, arm=arm, seed=seed,
            resume=resume))
                for cohort, arm, seed in development_cells(spec)]
    return [(spec["cohorts"][cohort]["gpu"], _command(
        spec_path, "full", cohort=cohort, arm=arm, seed=seed, resume=resume))
            for cohort, arm, seed in formal_cells(spec)]


def development_bank_commands(
        spec_path: Path, *, resume: bool = False
        ) -> list[tuple[int, list[str]]]:
    load_spec(spec_path)
    result = []
    for cohort in COHORTS:
        command = [
            str(ROOT / ".venv/bin/python"),
            "scripts/prepare_sage_mem_v1_development.py",
            "--spec", str(spec_path.resolve()), "--cohort", cohort,
            "--execute",
        ]
        if resume:
            command.append("--resume")
        result.append((-1, command))
    return result


def development_audit_command(spec_path: Path, *, resume: bool = False
                              ) -> tuple[int, list[str]]:
    command = [
        str(ROOT / ".venv/bin/python"), "scripts/audit_sage_mem_v1.py",
        "--stage", "development", "--spec", str(spec_path.resolve()),
        "--execute",
    ]
    if resume:
        command.append("--resume")
    return -1, command


def _environment(gpu: int) -> dict[str, str]:
    value = dict(os.environ)
    if gpu >= 0:
        value["CUDA_VISIBLE_DEVICES"] = str(gpu)
        value["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    value["PYTHONHASHSEED"] = "0"
    return value


def _command_label(command: list[str]) -> str:
    def option(name: str, fallback: str) -> str:
        return (command[command.index(name) + 1]
                if name in command else fallback)
    script = Path(command[1]).stem
    return "-".join((
        script, option("--cohort", "all"), option("--arm", "all"),
        f"seed-{option('--seed', 'all')}",
    ))


def _log_path(spec: dict, stage: str, gpu: int,
              command: list[str]) -> Path:
    owner = f"gpu-{gpu}" if gpu >= 0 else "cpu"
    return output_root(spec) / "logs" / stage / owner / (
        _command_label(command) + ".log")


def _run(gpu: int, command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    selected = log
    retry = 0
    while selected.exists():
        retry += 1
        selected = log.with_name(f"{log.stem}.retry-{retry}{log.suffix}")
    partial = selected.with_name(f".{selected.name}.partial-{os.getpid()}")
    with partial.open("x", encoding="utf-8") as stream:
        stream.write(f"physical_gpu\t{gpu if gpu >= 0 else 'cpu'}\n")
        stream.write("command\t" + " ".join(command) + "\n")
        stream.flush()
        result = subprocess.run(
            command, cwd=ROOT, env=_environment(gpu), stdout=stream,
            stderr=subprocess.STDOUT, text=True, check=False)
        stream.write(f"returncode\t{result.returncode}\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, selected)
    if result.returncode:
        raise RuntimeError(
            f"SAGE-Mem command failed ({result.returncode}); see {selected}")


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    spec = load_spec(args.spec)
    commands = planned_commands(args.spec, args.stage, resume=args.resume)
    if not args.execute:
        banks = (development_bank_commands(args.spec, resume=args.resume)
                 if args.stage == "development" else [])
        print(f"SAGE-Mem v1 preview: stage={args.stage} cells={len(commands)} "
              f"development_banks={len(banks)} "
              "physical_gpus={0,1,2}; no process launched")
        for gpu, command in banks:
            print(f"gpu=cpu bank {' '.join(command)}")
        for gpu, command in commands:
            print(f"gpu={gpu if gpu >= 0 else 'cpu'} {' '.join(command)}")
        if args.stage == "development":
            _, command = development_audit_command(
                args.spec, resume=args.resume)
            print("gpu=cpu audit " + " ".join(command))
        return
    if args.stage == "full" \
            and args.formal_confirmation != FORMAL_CONFIRMATION:
        raise RuntimeError(
            f"formal launcher requires --formal-confirmation {FORMAL_CONFIRMATION}")
    if args.stage in ("preflight", "seal"):
        gpu, command = commands[0]
        _run(gpu, command, _log_path(spec, args.stage, gpu, command))
        return
    if args.stage == "development":
        for gpu, command in development_bank_commands(
                args.spec, resume=args.resume):
            _run(gpu, command, _log_path(spec, "development-bank", gpu,
                                         command))
    # One serial queue per physical GPU. GPU 0 intentionally owns two cohorts.
    queues: dict[int, list[list[str]]] = {0: [], 1: [], 2: []}
    for gpu, command in commands:
        queues[gpu].append(command)

    def worker(gpu: int) -> None:
        for command in queues[gpu]:
            _run(gpu, command, _log_path(spec, args.stage, gpu, command))

    failures = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures: dict[Future[None], int] = {
            pool.submit(worker, gpu): gpu for gpu in queues if queues[gpu]}
        for future in as_completed(futures):
            try:
                future.result()
            except BaseException as error:
                failures.append((futures[future], str(error)))
    if failures:
        raise RuntimeError(f"SAGE-Mem GPU queues failed: {failures}")
    if args.stage == "development":
        gpu, command = development_audit_command(
            args.spec, resume=args.resume)
        _run(gpu, command, _log_path(
            spec, "development-audit", gpu, command))


if __name__ == "__main__":
    main()
