#!/usr/bin/env python3
"""Locked contract for the five-seed Reacher context/rollout extension."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "configs/paper_a_context_rollout_seed_extension_v1.yaml"
TASKS = ("t1", "t3")
TASK_NAMES = {
    "t1": "Transient-marker recall",
    "t3": "Drifting-color recall",
}
TASK_SLUGS = {
    "t1": "transient-marker-recall",
    "t3": "drifting-color-recall",
}
TASK_FAMILIES = {
    "t1": "transient-marker",
    "t3": "drifting-color",
}
CONTEXTS = (3, 16, 32, 56)
OBJECTIVES = ("one_step", "overshoot_8")
OBJECTIVE_NAMES = {
    "one_step": "One-step objective",
    "overshoot_8": "Eight-step overshooting",
}
PARENT_SEEDS = (0, 1, 2)
EXTENSION_SEEDS = (3, 4)
COMBINED_SEEDS = (0, 1, 2, 3, 4)
ALLOWED_DEVICES = ("cuda:1", "cuda:2")
FORBIDDEN_DEVICES = ("cuda:0", "cuda:3")


class ExtensionSpecError(ValueError):
    """Raised when the immutable extension contract is violated."""


@dataclass(frozen=True)
class ExtensionCell:
    wave: str
    task: str
    variant: str
    seed: int
    source: str
    metrics_path: Path

    @property
    def semantic_label(self) -> str:
        variant = (f"context {self.variant.removeprefix('h')}"
                   if self.wave == "long_context"
                   else OBJECTIVE_NAMES[self.variant])
        return f"{TASK_NAMES[self.task]} / {variant} / seed {self.seed}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ExtensionSpecError(f"{label} must be a mapping")
    return value


def _exact(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise ExtensionSpecError(
            f"{label} must be exactly {expected!r}, observed {value!r}")


def _finite_equal(value: Any, expected: float, label: str) -> None:
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not math.isclose(float(value), expected,
                                rel_tol=1e-12, abs_tol=1e-15)):
        raise ExtensionSpecError(
            f"{label} must be exactly {expected!r}, observed {value!r}")


def repo_path(value: Any, label: str, *, root: Path = ROOT) -> Path:
    if not isinstance(value, str) or not value:
        raise ExtensionSpecError(f"{label} must be a repository-relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ExtensionSpecError(f"{label} leaves the repository: {value!r}")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ExtensionSpecError(
            f"{label} leaves the repository: {value!r}") from error
    return candidate


def _artifact_record(value: Any, label: str, *, root: Path) -> Mapping[str, Any]:
    record = _mapping(value, label)
    repo_path(record.get("path"), f"{label}.path", root=root)
    digest = record.get("sha256")
    if (not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)):
        raise ExtensionSpecError(f"{label}.sha256 is not a lowercase SHA-256")
    return record


def validate_spec(spec: Mapping[str, Any], *, root: Path = ROOT) -> None:
    _exact(spec.get("schema_version"), 1, "schema_version")
    _exact(spec.get("study"),
           "paper-a-reacher-context-rollout-seed-extension-v1", "study")

    semantic = _mapping(spec.get("semantic_tasks"), "semantic_tasks")
    _exact(list(semantic), list(TASKS), "semantic_tasks keys")
    for task in TASKS:
        item = _mapping(semantic[task], f"semantic_tasks.{task}")
        _exact(item.get("name"), TASK_NAMES[task],
               f"semantic_tasks.{task}.name")
        _exact(item.get("slug"), TASK_SLUGS[task],
               f"semantic_tasks.{task}.slug")
        _exact(item.get("family"), TASK_FAMILIES[task],
               f"semantic_tasks.{task}.family")

    parent = _mapping(spec.get("parent"), "parent")
    output = _mapping(spec.get("output"), "output")
    seeds = _mapping(spec.get("seed_contract"), "seed_contract")
    context = _mapping(spec.get("long_context"), "long_context")
    rollout = _mapping(spec.get("learned_rollout"), "learned_rollout")
    analysis = _mapping(spec.get("analysis"), "analysis")
    execution = _mapping(spec.get("execution"), "execution")

    _exact(parent.get("study"), "paper-a-expansion", "parent.study")
    for key in ("config", "summary", "official_weights", "aggregator",
                "launcher"):
        _artifact_record(parent.get(key), f"parent.{key}", root=root)
    trainers = _mapping(parent.get("trainers"), "parent.trainers")
    _exact(set(trainers), {"long_context", "learned_rollout"},
           "parent.trainers keys")
    for key in trainers:
        _artifact_record(trainers[key], f"parent.trainers.{key}", root=root)
    caches = _mapping(parent.get("caches"), "parent.caches")
    _exact(set(caches), set(TASKS), "parent.caches keys")
    for task in TASKS:
        task_caches = _mapping(caches[task], f"parent.caches.{task}")
        _exact(set(task_caches), {"train", "validation"},
               f"parent.caches.{task} keys")
        for split in ("train", "validation"):
            _artifact_record(task_caches[split],
                             f"parent.caches.{task}.{split}", root=root)

    _exact(seeds.get("parent_seeds"), list(PARENT_SEEDS),
           "seed_contract.parent_seeds")
    _exact(seeds.get("extension_seeds"), list(EXTENSION_SEEDS),
           "seed_contract.extension_seeds")
    _exact(seeds.get("combined_seeds"), list(COMBINED_SEEDS),
           "seed_contract.combined_seeds")

    _exact(context.get("tasks"), list(TASKS), "long_context.tasks")
    _exact(context.get("contexts"), list(CONTEXTS), "long_context.contexts")
    context_exact = {
        "epochs": 60,
        "batch_size": 256,
        "grad_clip": 1.0,
        "num_workers": 0,
        "position_initialization": "interpolate",
        "amp": True,
        "amp_dtype": "bfloat16",
        "frozen_encoder": True,
        "trainable_components": ["action_encoder", "predictor", "pred_proj"],
        "objective": "final-token next-latent MSE over all exact-H windows",
        "decision_index": 63,
        "legal_input": "z[max(0,q-H):q] and actions[max(0,q-H):q]",
        "semantic_readout": (
            "final contextual prediction before decision observation q"),
    }
    for key, expected in context_exact.items():
        _exact(context.get(key), expected, f"long_context.{key}")
    _finite_equal(context.get("learning_rate"), 0.0001,
                  "long_context.learning_rate")
    _finite_equal(context.get("weight_decay"), 0.001,
                  "long_context.weight_decay")

    _exact(rollout.get("tasks"), list(TASKS), "learned_rollout.tasks")
    _exact(rollout.get("objectives"), list(OBJECTIVES),
           "learned_rollout.objectives")
    rollout_exact = {
        "objective_horizons": {"one_step": 1, "overshoot_8": 8},
        "epochs": 60,
        "batch_size": 64,
        "context": 3,
        "anchor": 24,
        "evaluation_horizons": [1, 2, 4, 8, 16],
        "frozen_encoder": True,
        "trainable_components": ["action_encoder", "predictor", "pred_proj"],
        "metrics": [
            "normalized_latent_mse", "copy_last_normalized_mse",
            "shuffled_action_normalized_mse", "true_action_advantage",
            "pose_angular_mae", "predicted_effective_rank",
            "target_effective_rank",
        ],
        "competence_gate": (
            "better than copy-last and positive true-action advantage at "
            "horizons 1, 2, 4, and 8"),
    }
    for key, expected in rollout_exact.items():
        _exact(rollout.get(key), expected, f"learned_rollout.{key}")
    _finite_equal(rollout.get("learning_rate"), 0.0001,
                  "learned_rollout.learning_rate")
    _finite_equal(rollout.get("weight_decay"), 0.001,
                  "learned_rollout.weight_decay")

    _exact(analysis.get("bootstrap_unit"), "matched optimizer/model seed",
           "analysis.bootstrap_unit")
    _exact(analysis.get("bootstrap_draws"), 20_000,
           "analysis.bootstrap_draws")
    _exact(analysis.get("bootstrap_seed"), 20_260_706,
           "analysis.bootstrap_seed")
    _finite_equal(analysis.get("confidence_level"), 0.95,
                  "analysis.confidence_level")
    _exact(analysis.get("interval"), "percentile", "analysis.interval")
    _exact(analysis.get("task_weighting"),
           "equal within matched seed for pooled summaries",
           "analysis.task_weighting")

    _exact(execution.get("allowed_devices"), list(ALLOWED_DEVICES),
           "execution.allowed_devices")
    _exact(execution.get("forbidden_devices"), list(FORBIDDEN_DEVICES),
           "execution.forbidden_devices")
    _exact(execution.get("default_gpus"), [1, 2],
           "execution.default_gpus")
    _exact(execution.get("max_jobs_per_gpu"), 1,
           "execution.max_jobs_per_gpu")
    _exact(execution.get("require_explicit_execute"), True,
           "execution.require_explicit_execute")

    parent_root = repo_path(parent.get("root"), "parent.root", root=root)
    output_root = repo_path(output.get("root"), "output.root", root=root)
    if parent_root == output_root or parent_root in output_root.parents:
        raise ExtensionSpecError("extension output overlaps the parent root")
    expected_output_keys = {
        "root", "long_context", "learned_rollout", "staging", "logs",
        "summary_json", "summary_markdown", "summary_sha256",
    }
    _exact(set(output), expected_output_keys, "output keys")
    for key in expected_output_keys.difference({"root"}):
        child = repo_path(output[key], f"output.{key}", root=root)
        if output_root not in child.parents:
            raise ExtensionSpecError(f"output.{key} is not below output.root")


def _lock_path(spec_path: Path) -> Path:
    return spec_path.with_suffix(".sha256")


def _verify_file_record(record: Mapping[str, Any], label: str,
                        *, root: Path) -> Path:
    path = repo_path(record["path"], f"{label}.path", root=root)
    if not path.is_file():
        raise ExtensionSpecError(f"missing {label}: {path}")
    observed = sha256_file(path)
    if observed != record["sha256"]:
        raise ExtensionSpecError(
            f"{label} hash mismatch: {observed} != {record['sha256']}")
    return path


def _verify_parent_grid(spec: Mapping[str, Any], *, root: Path) -> None:
    parent = spec["parent"]
    for key in ("config", "summary", "official_weights", "aggregator",
                "launcher"):
        _verify_file_record(parent[key], f"parent.{key}", root=root)
    for key, record in parent["trainers"].items():
        _verify_file_record(record, f"parent.trainers.{key}", root=root)
    for task in TASKS:
        for split in ("train", "validation"):
            _verify_file_record(parent["caches"][task][split],
                                f"parent.caches.{task}.{split}", root=root)

    summary_path = repo_path(parent["summary"]["path"],
                             "parent.summary.path", root=root)
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ExtensionSpecError(
            f"cannot read locked parent summary: {error}") from error
    completion = _mapping(summary.get("completion"), "parent summary completion")
    if completion.get("complete") is not True or completion.get("missing_count") != 0:
        raise ExtensionSpecError("locked parent summary is not complete")
    provenance = _mapping(summary.get("provenance"), "parent summary provenance")
    ledger = _mapping(provenance.get("source_metric_sha256"),
                      "parent summary metric ledger")
    for cell in expected_cells(spec, sources=("parent",), root=root):
        relative = str(cell.metrics_path.relative_to(root.resolve()))
        expected = ledger.get(relative)
        if not isinstance(expected, str):
            raise ExtensionSpecError(
                f"parent metric is absent from locked summary ledger: {relative}")
        if not cell.metrics_path.is_file():
            raise ExtensionSpecError(f"missing parent metric: {relative}")
        observed = sha256_file(cell.metrics_path)
        if observed != expected:
            raise ExtensionSpecError(
                f"parent metric hash mismatch for {relative}: "
                f"{observed} != {expected}")

    config_path = repo_path(parent["config"]["path"],
                            "parent.config.path", root=root)
    parent_config = _mapping(yaml.safe_load(config_path.read_text()),
                             "parent config")
    parent_context = _mapping(parent_config.get("long_context"),
                              "parent long_context")
    parent_rollout = _mapping(parent_config.get("learned_rollout"),
                              "parent learned_rollout")
    _exact(parent_context.get("tasks"), list(TASKS),
           "parent long_context.tasks")
    _exact(parent_context.get("contexts"), list(CONTEXTS),
           "parent long_context.contexts")
    _exact(parent_context.get("seeds"), list(PARENT_SEEDS),
           "parent long_context.seeds")
    for key, expected in (
            ("epochs", 60), ("batch_size", 256),
            ("learning_rate", 0.0001), ("weight_decay", 0.001),
            ("frozen_encoder", True), ("trainable_predictor", True)):
        _exact(parent_context.get(key), expected, f"parent long_context.{key}")
    _exact(parent_rollout.get("tasks"), list(TASKS),
           "parent learned_rollout.tasks")
    _exact(parent_rollout.get("objectives"), list(OBJECTIVES),
           "parent learned_rollout.objectives")
    _exact(parent_rollout.get("seeds"), list(PARENT_SEEDS),
           "parent learned_rollout.seeds")
    _exact(parent_rollout.get("epochs"), 60, "parent learned_rollout.epochs")
    _exact(parent_rollout.get("horizons"), [1, 2, 4, 8, 16],
           "parent learned_rollout.horizons")
    _exact(parent_rollout.get("frozen_encoder"), True,
           "parent learned_rollout.frozen_encoder")


def load_locked_spec(path: Path = DEFAULT_SPEC, *, verify_parent: bool = True,
                     root: Path = ROOT) -> dict[str, Any]:
    path = path.resolve()
    lock = _lock_path(path)
    if not path.is_file() or not lock.is_file():
        raise ExtensionSpecError(f"missing locked extension spec: {path}")
    tokens = lock.read_text().strip().split()
    if len(tokens) != 2 or tokens[1] != path.name:
        raise ExtensionSpecError(f"malformed extension lock file: {lock}")
    observed = sha256_file(path)
    if tokens[0] != observed:
        raise ExtensionSpecError(
            f"extension spec hash mismatch: {observed} != {tokens[0]}")
    try:
        payload = yaml.safe_load(path.read_text())
    except yaml.YAMLError as error:
        raise ExtensionSpecError(f"cannot parse extension spec: {error}") from error
    spec = dict(_mapping(payload, "extension spec"))
    validate_spec(spec, root=root)
    spec["_spec_record"] = {
        "path": str(path.relative_to(root.resolve())),
        "sha256": observed,
    }
    if verify_parent:
        _verify_parent_grid(spec, root=root)
    return spec


def resolve_path(spec: Mapping[str, Any], value: str,
                 *, root: Path = ROOT) -> Path:
    del spec  # Uniform call signature makes path use explicit at call sites.
    return repo_path(value, value, root=root)


def task_record(spec: Mapping[str, Any], task: str) -> Mapping[str, Any]:
    if task not in TASKS:
        raise ExtensionSpecError(f"unknown internal task key {task!r}")
    return _mapping(spec["semantic_tasks"][task], f"semantic_tasks.{task}")


def validate_device(spec: Mapping[str, Any], device: str) -> str:
    if device in spec["execution"]["forbidden_devices"]:
        raise ExtensionSpecError(f"device {device} is explicitly forbidden")
    if device not in spec["execution"]["allowed_devices"]:
        raise ExtensionSpecError(
            f"device {device} is not in the locked allowed-device set")
    return device


def extension_directory(spec: Mapping[str, Any], wave: str, task: str,
                        variant: str, seed: int,
                        *, root: Path = ROOT) -> Path:
    if wave == "long_context":
        if variant not in {f"h{history}" for history in CONTEXTS}:
            raise ExtensionSpecError(f"invalid context variant {variant!r}")
        base = repo_path(spec["output"]["long_context"],
                         "output.long_context", root=root)
    elif wave == "learned_rollout":
        if variant not in OBJECTIVES:
            raise ExtensionSpecError(f"invalid rollout objective {variant!r}")
        base = repo_path(spec["output"]["learned_rollout"],
                         "output.learned_rollout", root=root)
    else:
        raise ExtensionSpecError(f"unknown wave {wave!r}")
    task_record(spec, task)
    if seed not in EXTENSION_SEEDS:
        raise ExtensionSpecError(f"seed {seed} is outside the extension deck")
    return base / task / variant / f"s{seed}"


def expected_cells(spec: Mapping[str, Any], *,
                   sources: Sequence[str] = ("parent", "extension"),
                   root: Path = ROOT) -> list[ExtensionCell]:
    allowed_sources = {"parent", "extension"}
    if not sources or not set(sources).issubset(allowed_sources):
        raise ExtensionSpecError(f"invalid cell sources {sources!r}")
    parent_root = repo_path(spec["parent"]["root"], "parent.root", root=root)
    cells: list[ExtensionCell] = []
    for source in sources:
        seeds = PARENT_SEEDS if source == "parent" else EXTENSION_SEEDS
        for task in TASKS:
            for history in CONTEXTS:
                variant = f"h{history}"
                for seed in seeds:
                    if source == "parent":
                        metrics = (parent_root / "long_context" / task / variant
                                   / f"s{seed}" / "metrics.json")
                    else:
                        metrics = extension_directory(
                            spec, "long_context", task, variant, seed,
                            root=root) / "metrics.json"
                    cells.append(ExtensionCell(
                        "long_context", task, variant, seed, source, metrics))
            for objective in OBJECTIVES:
                for seed in seeds:
                    if source == "parent":
                        metrics = (parent_root / "rollout" / task / objective
                                   / f"s{seed}" / "metrics.json")
                    else:
                        metrics = extension_directory(
                            spec, "learned_rollout", task, objective, seed,
                            root=root) / "metrics.json"
                    cells.append(ExtensionCell(
                        "learned_rollout", task, objective, seed,
                        source, metrics))
    labels = [(cell.wave, cell.task, cell.variant, cell.seed) for cell in cells]
    if len(labels) != len(set(labels)):
        raise ExtensionSpecError("combined grid contains duplicate cells")
    return cells
