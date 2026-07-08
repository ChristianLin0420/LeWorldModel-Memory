#!/usr/bin/env python3
"""Validate the locked Paper-A Reacher robustness specification."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "configs/paper_a_robustness.yaml"
TASKS = ("t1", "t3")
SEED_EXTENSION_ARMS = ("gru", "ssm", "fixed_trust")
PARENT_ARMS = ("none", "gru", "lstm", "ssm", "fixed_trust")
PARENT_SEEDS = (0, 1, 2, 3, 4)
EXTENSION_SEEDS = (5, 6, 7, 8, 9)
ALLOWED_DEVICES = ("cuda:1", "cuda:2")
FORBIDDEN_DEVICES = ("cuda:0", "cuda:3")


class RobustnessSpecError(ValueError):
    """A fail-closed strengthening-spec validation error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise RobustnessSpecError(f"{label} must be a mapping")
    return value


def _repo_path(value: Any, label: str, *, root: Path = ROOT) -> Path:
    if not isinstance(value, str) or not value:
        raise RobustnessSpecError(f"{label} must be a repository-relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise RobustnessSpecError(f"{label} leaves the repository: {value!r}")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise RobustnessSpecError(
            f"{label} leaves the repository: {value!r}") from error
    return candidate


def _exact_sequence(value: Any, expected: tuple[Any, ...], label: str) -> None:
    if value != list(expected):
        raise RobustnessSpecError(
            f"{label} must be exactly {list(expected)!r}, observed {value!r}")


def validate_spec(spec: Mapping[str, Any], *, root: Path = ROOT) -> None:
    if spec.get("schema_version") != 1:
        raise RobustnessSpecError("schema_version must be 1")
    if spec.get("study") != "paper-a-reacher-robustness-v1":
        raise RobustnessSpecError("unexpected robustness study identity")

    parent = _mapping(spec.get("parent"), "parent")
    output = _mapping(spec.get("output"), "output")
    fresh = _mapping(spec.get("fresh_validation"), "fresh_validation")
    extension = _mapping(
        spec.get("carrier_seed_extension"), "carrier_seed_extension")
    execution = _mapping(spec.get("execution"), "execution")

    _exact_sequence(spec.get("tasks"), TASKS, "tasks")
    _exact_sequence(fresh.get("checkpoint_arms"), PARENT_ARMS,
                    "fresh_validation.checkpoint_arms")
    _exact_sequence(fresh.get("checkpoint_seeds"), PARENT_SEEDS,
                    "fresh_validation.checkpoint_seeds")
    _exact_sequence(extension.get("arms"), SEED_EXTENSION_ARMS,
                    "carrier_seed_extension.arms")
    _exact_sequence(extension.get("seeds"), EXTENSION_SEEDS,
                    "carrier_seed_extension.seeds")
    _exact_sequence(execution.get("allowed_devices"), ALLOWED_DEVICES,
                    "execution.allowed_devices")
    _exact_sequence(execution.get("forbidden_devices"), FORBIDDEN_DEVICES,
                    "execution.forbidden_devices")
    if execution.get("default_device") not in ALLOWED_DEVICES:
        raise RobustnessSpecError("default_device is not allowed")
    if execution.get("require_explicit_execute") is not True:
        raise RobustnessSpecError("the robustness launcher must require --execute")

    if fresh.get("episodes_per_bank") != 240:
        raise RobustnessSpecError("fresh validation banks must contain 240 episodes")
    if float(fresh.get("categorical_availability_min", -1)) != 0.75:
        raise RobustnessSpecError("fresh validation availability gate changed")
    banks = fresh.get("banks")
    if not isinstance(banks, list) or len(banks) != 2:
        raise RobustnessSpecError("fresh_validation.banks must contain two banks")
    bank_ids, bank_seeds = [], []
    for bank in banks:
        item = _mapping(bank, "fresh_validation bank")
        bank_ids.append(item.get("id"))
        bank_seeds.append(item.get("collection_seed"))
    if bank_ids != ["fresh-a", "fresh-b"]:
        raise RobustnessSpecError("fresh validation bank identities changed")
    if bank_seeds != [270703, 270704] or len(set(bank_seeds)) != len(bank_seeds):
        raise RobustnessSpecError("fresh validation collection seeds changed")

    expected_training = {
        "epochs": 100,
        "batch_size": 64,
        "learning_rate": 0.0003,
        "weight_decay": 0.00001,
        "train_cache_source": "parent",
        "validation_cache_source": "parent",
    }
    for key, expected in expected_training.items():
        if extension.get(key) != expected:
            raise RobustnessSpecError(
                f"carrier_seed_extension.{key} must be {expected!r}")

    parent_root = _repo_path(parent.get("root"), "parent.root", root=root)
    robustness_root = _repo_path(output.get("root"), "output.root", root=root)
    if parent_root == robustness_root or parent_root in robustness_root.parents:
        raise RobustnessSpecError(
            "robustness output must not equal or lie below the parent artifact root")
    for key in (
            "fresh_validation_data", "fresh_validation_cache",
            "fresh_validation_evaluation", "carrier_seed_extension", "logs"):
        child = _repo_path(output.get(key), f"output.{key}", root=root)
        if robustness_root not in child.parents:
            raise RobustnessSpecError(
                f"output.{key} must lie strictly below output.root")

    for label, record in (
            ("parent.config", parent.get("config")),
            ("parent.summary", parent.get("summary")),
            ("parent.official_weights", parent.get("official_weights"))):
        item = _mapping(record, label)
        _repo_path(item.get("path"), f"{label}.path", root=root)
        digest = item.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise RobustnessSpecError(f"{label}.sha256 is invalid")
    for split in ("train_caches", "validation_caches"):
        records = _mapping(parent.get(split), f"parent.{split}")
        if set(records) != set(TASKS):
            raise RobustnessSpecError(f"parent.{split} must cover {TASKS}")
        for task in TASKS:
            item = _mapping(records[task], f"parent.{split}.{task}")
            _repo_path(item.get("path"), f"parent.{split}.{task}.path", root=root)
            digest = item.get("sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                raise RobustnessSpecError(
                    f"parent.{split}.{task}.sha256 is invalid")


def _lock_path(spec_path: Path) -> Path:
    return spec_path.with_suffix(".sha256")


def load_locked_spec(path: Path = DEFAULT_SPEC, *, verify_parent: bool = True,
                     root: Path = ROOT) -> dict[str, Any]:
    path = path.resolve()
    lock = _lock_path(path)
    if not path.is_file() or not lock.is_file():
        raise RobustnessSpecError(f"missing locked robustness spec: {path}")
    tokens = lock.read_text().strip().split()
    if len(tokens) != 2 or tokens[1] != path.name:
        raise RobustnessSpecError(f"malformed robustness lock file: {lock}")
    actual = sha256_file(path)
    if tokens[0] != actual:
        raise RobustnessSpecError(
            f"robustness spec hash mismatch: {actual} != {tokens[0]}")
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as error:
        raise RobustnessSpecError(f"cannot parse robustness spec: {error}") from error
    spec = dict(_mapping(raw, "robustness spec"))
    validate_spec(spec, root=root)
    spec["_spec_record"] = {
        "path": str(path.relative_to(root.resolve())),
        "sha256": actual,
    }
    if verify_parent:
        parent = spec["parent"]
        records = [parent["config"], parent["summary"],
                   parent["official_weights"]]
        records.extend(parent["train_caches"].values())
        records.extend(parent["validation_caches"].values())
        for record in records:
            source = _repo_path(record["path"], record["path"], root=root)
            if not source.is_file():
                raise RobustnessSpecError(f"missing parent artifact: {source}")
            digest = sha256_file(source)
            if digest != record["sha256"]:
                raise RobustnessSpecError(
                    f"parent artifact hash mismatch for {record['path']}: "
                    f"{digest} != {record['sha256']}")
    return spec


def bank_by_id(spec: Mapping[str, Any], bank_id: str) -> Mapping[str, Any]:
    for bank in spec["fresh_validation"]["banks"]:
        if bank["id"] == bank_id:
            return bank
    raise RobustnessSpecError(f"unknown fresh validation bank {bank_id!r}")


def resolve_spec_path(spec: Mapping[str, Any], value: str,
                      *, root: Path = ROOT) -> Path:
    return _repo_path(value, value, root=root)


def validate_device(spec: Mapping[str, Any], device: str,
                    *, allow_cpu: bool = False) -> None:
    if allow_cpu and device == "cpu":
        return
    if device in spec["execution"]["forbidden_devices"]:
        raise RobustnessSpecError(f"device {device} is explicitly forbidden")
    if device not in spec["execution"]["allowed_devices"]:
        raise RobustnessSpecError(
            f"device {device} is not in the allowed robustness device set")
