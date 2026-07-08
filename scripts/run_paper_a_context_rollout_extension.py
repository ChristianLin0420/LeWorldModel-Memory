#!/usr/bin/env python3
"""Run one isolated, provenance-receipted context/rollout extension cell.

The underlying trainers are the byte-pinned parent programs.  This wrapper
only validates the locked seed deck, stages the output outside the parent
namespace, hashes the three trainer products, and atomically publishes the
completed cell with a provenance receipt.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.paper_a_context_rollout_extension_spec import (  # noqa: E402
    CONTEXTS,
    DEFAULT_SPEC,
    EXTENSION_SEEDS,
    OBJECTIVES,
    TASKS,
    ExtensionSpecError,
    extension_directory,
    load_locked_spec,
    repo_path,
    sha256_file,
    task_record,
    validate_device,
)


PYTHON = sys.executable
WAVES = ("long_context", "learned_rollout")
PRODUCTS = ("metrics.json", "history.csv", "checkpoint.pt")


def cell_variant(wave: str, context: int | None,
                 objective: str | None) -> str:
    if wave == "long_context":
        if context not in CONTEXTS or objective is not None:
            raise ExtensionSpecError(
                "long_context requires one locked --context and no --objective")
        return f"h{context}"
    if wave == "learned_rollout":
        if objective not in OBJECTIVES or context is not None:
            raise ExtensionSpecError(
                "learned_rollout requires one locked --objective and no --context")
        return str(objective)
    raise ExtensionSpecError(f"unknown extension wave {wave!r}")


def stage_paths(spec: Mapping[str, Any], wave: str, task: str,
                variant: str, seed: int) -> tuple[Path, Path, Path]:
    staging = repo_path(spec["output"]["staging"], "output.staging")
    name = f"{wave}-{task}-{variant}-s{seed}"
    job_stage = staging / name
    if wave == "long_context":
        produced = job_stage / "cell"
    else:
        produced = job_stage / "root" / task / variant / f"s{seed}"
    final = extension_directory(spec, wave, task, variant, seed)
    return job_stage, produced, final


def underlying_command(spec: Mapping[str, Any], wave: str, task: str,
                       variant: str, seed: int, device: str,
                       produced: Path) -> tuple[str, ...]:
    validate_device(spec, device)
    if task not in TASKS or seed not in EXTENSION_SEEDS:
        raise ExtensionSpecError("cell lies outside the locked extension grid")
    parent = spec["parent"]
    weights = repo_path(parent["official_weights"]["path"],
                        "parent.official_weights.path")
    cache_root = repo_path(parent["cache_root"], "parent.cache_root")
    if wave == "long_context":
        history = int(variant.removeprefix("h"))
        if history not in CONTEXTS or variant != f"h{history}":
            raise ExtensionSpecError(f"invalid context variant {variant!r}")
        deck = spec["long_context"]
        cache = parent["caches"][task]
        return (
            PYTHON, "scripts/train_official_long_context.py",
            "--train-cache", str(repo_path(
                cache["train"]["path"], f"parent.caches.{task}.train.path")),
            "--val-cache", str(repo_path(
                cache["validation"]["path"],
                f"parent.caches.{task}.validation.path")),
            "--official-checkpoint", str(weights),
            "--output-dir", str(produced),
            "--history-len", str(history),
            "--position-init", str(deck["position_initialization"]),
            "--task-family", str(task_record(spec, task)["family"]),
            "--epochs", str(deck["epochs"]),
            "--batch-size", str(deck["batch_size"]),
            "--lr", str(deck["learning_rate"]),
            "--weight-decay", str(deck["weight_decay"]),
            "--grad-clip", str(deck["grad_clip"]),
            "--num-workers", str(deck["num_workers"]),
            "--seed", str(seed), "--device", device,
            "--amp", "--amp-dtype", str(deck["amp_dtype"]),
        )
    if wave == "learned_rollout":
        if variant not in OBJECTIVES:
            raise ExtensionSpecError(f"invalid rollout objective {variant!r}")
        deck = spec["learned_rollout"]
        # The rollout trainer appends task/objective/seed below --output.
        stage_root = produced.parents[2]
        return (
            PYTHON, "scripts/train_official_rollout.py",
            "--task", task, "--objective", variant, "--seed", str(seed),
            "--cache-root", str(cache_root), "--weights", str(weights),
            "--output", str(stage_root), "--epochs", str(deck["epochs"]),
            "--batch-size", str(deck["batch_size"]),
            "--lr", str(deck["learning_rate"]),
            "--weight-decay", str(deck["weight_decay"]),
            "--device", device,
        )
    raise ExtensionSpecError(f"unknown extension wave {wave!r}")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_metrics(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"trainer produced unreadable metrics: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError("trainer metrics root is not an object")
    return payload


def _basic_product_check(wave: str, task: str, variant: str, seed: int,
                         directory: Path) -> Mapping[str, Any]:
    missing = [name for name in PRODUCTS if not (directory / name).is_file()]
    if missing:
        raise RuntimeError(f"trainer omitted required products: {missing}")
    metrics = _load_metrics(directory / "metrics.json")
    if wave == "long_context":
        config = metrics.get("config", {})
        expected_history = int(variant.removeprefix("h"))
        valid = (metrics.get("study") == "official-lewm-long-context"
                 and config.get("seed") == seed
                 and config.get("history_len") == expected_history)
    else:
        valid = (metrics.get("study") == "official-lewm-learned-rollout"
                 and metrics.get("task") == task
                 and metrics.get("objective") == variant
                 and metrics.get("seed") == seed)
    if not valid:
        raise RuntimeError("trainer metrics do not identify the requested cell")
    return metrics


def build_receipt(spec: Mapping[str, Any], wave: str, task: str,
                  variant: str, seed: int, device: str,
                  command: tuple[str, ...], directory: Path) -> dict[str, Any]:
    _basic_product_check(wave, task, variant, seed, directory)
    trainer_key = "long_context" if wave == "long_context" else "learned_rollout"
    return {
        "schema_version": 1,
        "study": spec["study"],
        "spec": dict(spec["_spec_record"]),
        "source": "new extension training cell",
        "wave": wave,
        "semantic_task_name": task_record(spec, task)["name"],
        "semantic_task_slug": task_record(spec, task)["slug"],
        "internal_task_key": task,
        "variant": variant,
        "variant_name": (f"Context length {variant.removeprefix('h')}"
                         if wave == "long_context" else
                         ("One-step objective" if variant == "one_step"
                          else "Eight-step overshooting")),
        "seed": seed,
        "device": device,
        "parent_config": dict(spec["parent"]["config"]),
        "official_weights": dict(spec["parent"]["official_weights"]),
        "trainer": dict(spec["parent"]["trainers"][trainer_key]),
        "command": list(command),
        "products": {
            name: {"sha256": sha256_file(directory / name),
                   "bytes": (directory / name).stat().st_size}
            for name in PRODUCTS
        },
        "parent_artifacts_modified": False,
    }


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.setdefault("PYTHONHASHSEED", "0")
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                     "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment.setdefault(variable, "1")
    return environment


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--wave", required=True, choices=WAVES)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--context", type=int, choices=CONTEXTS)
    parser.add_argument("--objective", choices=OBJECTIVES)
    parser.add_argument("--seed", required=True, type=int,
                        choices=EXTENSION_SEEDS)
    parser.add_argument("--device", required=True)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    spec = load_locked_spec(args.spec)
    validate_device(spec, args.device)
    variant = cell_variant(args.wave, args.context, args.objective)
    job_stage, produced, final = stage_paths(
        spec, args.wave, args.task, variant, args.seed)
    command = underlying_command(
        spec, args.wave, args.task, variant, args.seed, args.device, produced)
    print(f"[context-rollout-cell] {task_record(spec, args.task)['name']} / "
          f"{variant} / seed {args.seed} / {args.device} / "
          f"execute={args.execute}", flush=True)
    if not args.execute:
        print(shlex.join(command))
        return
    if final.exists():
        raise FileExistsError(f"refusing to overwrite completed cell: {final}")
    if job_stage.exists():
        raise FileExistsError(
            f"staging directory exists from an earlier attempt: {job_stage}")
    job_stage.mkdir(parents=True)
    result = subprocess.run(command, cwd=ROOT, env=_environment(), check=False)
    if result.returncode != 0:
        raise SystemExit(
            f"underlying trainer failed with exit {result.returncode}; "
            f"staging retained at {job_stage}")
    receipt = build_receipt(
        spec, args.wave, args.task, variant, args.seed,
        args.device, command, produced)
    _atomic_json(produced / "receipt.json", receipt)
    final.parent.mkdir(parents=True, exist_ok=True)
    produced.rename(final)
    # Remove only empty staging parents created for this new cell.
    cursor = produced.parent
    staging_root = repo_path(spec["output"]["staging"], "output.staging")
    while cursor != staging_root.parent and cursor.exists():
        try:
            cursor.rmdir()
        except OSError:
            break
        if cursor == staging_root:
            break
        cursor = cursor.parent
    print(f"[context-rollout-cell] committed {final}", flush=True)


if __name__ == "__main__":
    main()
