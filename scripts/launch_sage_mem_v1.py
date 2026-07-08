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


def formal_audit_command(spec_path: Path, *, resume: bool = False
                         ) -> tuple[int, list[str]]:
    command = [
        str(ROOT / ".venv/bin/python"), "scripts/audit_sage_mem_v1.py",
        "--stage", "formal", "--spec", str(spec_path.resolve()),
        "--execute",
    ]
    if resume:
        command.append("--resume")
    return -1, command


def formal_preparation_command(spec_path: Path) -> tuple[int, list[str]]:
    from scripts.prepare_sage_mem_v1_formal import CONFIRMATION

    return -1, [
        str(ROOT / ".venv/bin/python"),
        "scripts/prepare_sage_mem_v1_formal.py",
        "--stage", "prepare", "--spec", str(spec_path.resolve()),
        "--confirmation", CONFIRMATION,
    ]


def formal_raw_context_command(
        spec_path: Path, spec: dict, *, resume: bool = False
        ) -> tuple[int, list[str]]:
    root = output_root(spec)
    command = [
        str(ROOT / ".venv/bin/python"),
        "scripts/prepare_sage_mem_v1_raw_context_reference.py",
        "--config", str(spec_path.resolve()),
        "--prepared-root", str(root / "formal_preparation" / "banks"),
        "--output-root", str(root / "raw_context_phase_a"),
        "--execute",
    ]
    if resume:
        command.append("--resume")
    return -1, command


def formal_execution_decks_command(
        spec_path: Path, spec: dict, *, resume: bool = False
        ) -> tuple[int, list[str]]:
    root = output_root(spec)
    command = [
        str(ROOT / ".venv/bin/python"),
        "scripts/prepare_sage_mem_v1_execution_decks.py",
        "--spec", str(spec_path.resolve()),
        "--preparation-root", str(root / "formal_preparation"),
        "--output-root", str(
            root / "formal_preparation" / "execution_decks"),
        "--execute",
    ]
    if resume:
        command.append("--resume")
    return -1, command


def formal_finalization_command(spec_path: Path, spec: dict, *,
                                validate_existing: bool = False) \
        -> tuple[int, list[str]]:
    root = output_root(spec)
    command = [
        str(ROOT / ".venv/bin/python"),
        "scripts/sage_mem_v1_formal_finalizer.py",
        "--phase-a-root", str(root),
        "--label-registry", str(
            root / "formal_preparation" / "custody" / "registry.json"),
        "--output-root", str(root / "formal_finalized"),
        "--execute",
    ]
    raw_context = root / "raw_context_phase_a"
    if raw_context.is_dir():
        command.extend(("--raw-context-root", str(raw_context)))
    execution_decks = (root / "formal_preparation" / "execution_decks"
                       / "registry.json")
    if execution_decks.is_file():
        command.extend(("--execution-deck-registry", str(execution_decks)))
    if validate_existing:
        command.append("--validate-finalized-output")
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
        if args.stage == "prepare":
            _, command = formal_preparation_command(args.spec)
            print("gpu=coordinator formal-banks " + " ".join(command))
            _, command = formal_raw_context_command(
                args.spec, spec, resume=args.resume)
            print("gpu=cpu raw-context " + " ".join(command))
            _, command = formal_execution_decks_command(
                args.spec, spec, resume=args.resume)
            print("gpu=cpu execution-decks " + " ".join(command))
        for gpu, command in commands:
            print(f"gpu={gpu if gpu >= 0 else 'cpu'} {' '.join(command)}")
        if args.stage == "development":
            _, command = development_audit_command(
                args.spec, resume=args.resume)
            print("gpu=cpu audit " + " ".join(command))
        if args.stage == "full":
            _, command = formal_finalization_command(args.spec, spec)
            print("gpu=cpu formal-finalizer " + " ".join(command))
            _, command = formal_audit_command(
                args.spec, resume=args.resume)
            print("gpu=cpu formal-audit " + " ".join(command))
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
    if args.stage == "prepare":
        gpu, command = formal_preparation_command(args.spec)
        _run(gpu, command, _log_path(
            spec, "formal-preparation", gpu, command))
        gpu, command = formal_raw_context_command(
            args.spec, spec, resume=args.resume)
        _run(gpu, command, _log_path(
            spec, "formal-raw-context", gpu, command))
        gpu, command = formal_execution_decks_command(
            args.spec, spec, resume=args.resume)
        _run(gpu, command, _log_path(
            spec, "formal-execution-decks", gpu, command))
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
    if args.stage == "full":
        destination = output_root(spec) / "formal_finalized" / "summary.json"
        gpu, command = formal_finalization_command(
            args.spec, spec,
            validate_existing=bool(destination.exists() and args.resume))
        _run(gpu, command, _log_path(
            spec, "formal-finalization", gpu, command))
        gpu, command = formal_audit_command(
            args.spec, resume=args.resume)
        _run(gpu, command, _log_path(
            spec, "formal-audit", gpu, command))


if __name__ == "__main__":
    main()
