#!/usr/bin/env python3
"""Fail-closed loader for the Paper-A evidence-age protocol."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "configs/paper_a_evidence_age_v1.yaml"
DEFAULT_SHA = ROOT / "configs/paper_a_evidence_age_v1.sha256"
DEFAULT_CODE_LOCK = ROOT / "configs/paper_a_evidence_age_v1.lock.json"
HOSTS = ("reacher", "pusht")
ARMS = ("none", "gru", "lstm", "ssm", "fixed_trust")
SEEDS = (0, 1, 2, 3, 4)
ALLOWED_DEVICES = ("cuda:0",)
REACHER_TASKS = ("t1", "t3")
PUSHT_TASKS = (
    "transient-visual-token-recall",
    "multi-item-visual-binding-recall",
)


class EvidenceAgeSpecError(ValueError):
    """The immutable evidence-age contract is malformed or changed."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(value: Any, *, root: Path = ROOT) -> Path:
    if not isinstance(value, str) or not value:
        raise EvidenceAgeSpecError("artifact path must be repository-relative")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise EvidenceAgeSpecError(f"artifact path leaves repository: {value!r}")
    result = (root / relative).resolve()
    try:
        result.relative_to(root.resolve())
    except ValueError as error:
        raise EvidenceAgeSpecError(
            f"artifact path leaves repository: {value!r}") from error
    return result


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceAgeSpecError(f"{label} must be a mapping")
    return value


def _exact(value: Any, expected: tuple[Any, ...], label: str) -> None:
    if value != list(expected):
        raise EvidenceAgeSpecError(
            f"{label} must be exactly {list(expected)!r}, got {value!r}")


def _verify_record(record: Any, label: str, *, root: Path) -> None:
    item = _mapping(record, label)
    path = resolve_path(item.get("path"), root=root)
    expected = item.get("sha256")
    if not path.is_file() or not isinstance(expected, str) \
            or sha256_file(path) != expected:
        raise EvidenceAgeSpecError(f"{label} identity mismatch: {path}")


def validate_spec(spec: Mapping[str, Any], *, root: Path = ROOT,
                  verify_parents: bool = True) -> None:
    if spec.get("schema_version") != 1 \
            or spec.get("study") != "paper-a-evidence-age-v1" \
            or spec.get("protocol_status") != "locked-before-metrics":
        raise EvidenceAgeSpecError("unexpected evidence-age study identity")
    if spec.get("implementation_lock") \
            != "configs/paper_a_evidence_age_v1.lock.json":
        raise EvidenceAgeSpecError("implementation lock path changed")
    _exact(spec.get("arms"), ARMS, "arms")
    _exact(spec.get("seeds"), SEEDS, "seeds")

    execution = _mapping(spec.get("execution"), "execution")
    _exact(execution.get("allowed_devices"), ALLOWED_DEVICES,
           "execution.allowed_devices")
    if execution.get("default_device") != "cuda:0" \
            or execution.get("explicit_execute_required") is not True \
            or execution.get("deterministic_algorithms") is not True:
        raise EvidenceAgeSpecError("execution contract changed")

    read = _mapping(spec.get("read_time"), "read_time")
    _exact(read.get("reacher_ages"), (4, 8, 15, 24, 32, 40, "final"),
           "read_time.reacher_ages")
    _exact(read.get("pusht_ages"), (4, 8, 15),
           "read_time.pusht_ages")
    if read.get("context_history") != 3 \
            or read.get("current_observation_excluded") is not True:
        raise EvidenceAgeSpecError("read-time causal endpoint changed")

    strict = _mapping(spec.get("strict_fixed_endpoint"),
                      "strict_fixed_endpoint")
    _exact(_mapping(strict.get("reacher"), "strict.reacher").get("ages"),
           (15, 24, 32, 43), "strict.reacher.ages")
    _exact(_mapping(strict.get("pusht"), "strict.pusht").get("ages"),
           (4, 8, 15), "strict.pusht.ages")
    training = _mapping(strict.get("carrier_training"),
                        "strict.carrier_training")
    expected_training = {
        "epochs": 100, "batch_size": 64, "learning_rate": 0.0003,
        "weight_decay": 0.00001, "windows_per_episode": 8,
        "frozen_encoder": True, "frozen_predictor": True,
    }
    if any(training.get(key) != value
           for key, value in expected_training.items()):
        raise EvidenceAgeSpecError("strict carrier-training contract changed")
    admission = _mapping(strict.get("admission"), "strict.admission")
    if float(admission.get("cue_accuracy_min", -1)) != 0.75 \
            or float(admission.get("shortcut_margin_above_chance", -1)) != 0.05:
        raise EvidenceAgeSpecError("strict admission thresholds changed")

    statistics = _mapping(spec.get("statistics"), "statistics")
    if statistics.get("bootstrap_draws") != 20000 \
            or statistics.get("bootstrap_seed") != 20260711:
        raise EvidenceAgeSpecError("bootstrap contract changed")

    parents = _mapping(spec.get("parents"), "parents")
    reacher = _mapping(parents.get("reacher"), "parents.reacher")
    pusht = _mapping(parents.get("pusht"), "parents.pusht")
    if tuple(item.get("key") for item in reacher.get("tasks", [])) \
            != REACHER_TASKS:
        raise EvidenceAgeSpecError("Reacher tasks changed")
    if tuple(item.get("key") for item in pusht.get("tasks", [])) \
            != PUSHT_TASKS:
        raise EvidenceAgeSpecError("PushT tasks changed")
    if verify_parents:
        for label, record in (
                ("parents.reacher.config", reacher.get("config")),
                ("parents.reacher.weights", reacher.get("weights")),
                ("parents.pusht.config", pusht.get("config")),
                ("parents.pusht.lock", pusht.get("lock")),
                ("parents.pusht.weights", pusht.get("weights"))):
            _verify_record(record, label, root=root)

    outputs = _mapping(spec.get("outputs"), "outputs")
    output_root = resolve_path(outputs.get("root"), root=root)
    for key in ("read_time", "strict", "logs"):
        child = resolve_path(outputs.get(key), root=root)
        if output_root not in child.parents:
            raise EvidenceAgeSpecError(f"outputs.{key} is outside output root")


def _verify_sha_sidecar(spec_path: Path, sha_path: Path) -> str:
    if not sha_path.is_file():
        raise EvidenceAgeSpecError(f"missing protocol SHA sidecar: {sha_path}")
    fields = sha_path.read_text().strip().split()
    if len(fields) != 2 or fields[1] != spec_path.name:
        raise EvidenceAgeSpecError("malformed protocol SHA sidecar")
    actual = sha256_file(spec_path)
    if fields[0] != actual:
        raise EvidenceAgeSpecError("protocol YAML differs from locked SHA")
    return actual


def _verify_implementation_lock(spec: Mapping[str, Any], spec_digest: str,
                                *, root: Path = ROOT) -> dict[str, Any]:
    path = resolve_path(spec.get("implementation_lock"), root=root)
    if not path.is_file():
        raise EvidenceAgeSpecError(f"missing implementation lock: {path}")
    import json
    value = json.loads(path.read_text())
    if value.get("schema_version") != 1 \
            or value.get("study") != spec.get("study") \
            or value.get("spec_sha256") != spec_digest:
        raise EvidenceAgeSpecError("implementation lock identity mismatch")
    producers = value.get("producers")
    if not isinstance(producers, dict) or not producers:
        raise EvidenceAgeSpecError("implementation lock has no producers")
    for relative, expected in producers.items():
        producer = resolve_path(relative, root=root)
        if not producer.is_file() or sha256_file(producer) != expected:
            raise EvidenceAgeSpecError(
                f"locked producer changed or disappeared: {relative}")
    return {
        "path": str(path.relative_to(root)),
        "sha256": sha256_file(path),
        "producers": len(producers),
    }


def load_locked_spec(spec_path: Path = DEFAULT_SPEC,
                     sha_path: Path = DEFAULT_SHA, *,
                     verify_parents: bool = True) -> dict[str, Any]:
    spec_path = spec_path.resolve()
    sha_path = sha_path.resolve()
    digest = _verify_sha_sidecar(spec_path, sha_path)
    value = yaml.safe_load(spec_path.read_text())
    if not isinstance(value, dict):
        raise EvidenceAgeSpecError("protocol YAML must contain one mapping")
    spec = dict(value)
    validate_spec(spec, verify_parents=verify_parents)
    implementation = _verify_implementation_lock(spec, digest)
    spec["_lock"] = {
        "path": str(spec_path.relative_to(ROOT)), "sha256": digest,
        "sidecar": str(sha_path.relative_to(ROOT)),
        "sidecar_sha256": sha256_file(sha_path),
        "implementation": implementation,
    }
    return spec


def validate_device(spec: Mapping[str, Any], device: str) -> str:
    if device not in spec["execution"]["allowed_devices"]:
        raise EvidenceAgeSpecError(
            f"device {device!r} forbidden; only cuda:0 is admitted")
    return device


def host_tasks(spec: Mapping[str, Any], host: str) -> tuple[str, ...]:
    if host not in HOSTS:
        raise EvidenceAgeSpecError(f"unknown host {host!r}")
    return tuple(item["key"] for item in spec["parents"][host]["tasks"])


def output_root(spec: Mapping[str, Any], branch: str) -> Path:
    if branch not in ("read_time", "strict", "logs"):
        raise EvidenceAgeSpecError(f"unknown output branch {branch!r}")
    return resolve_path(spec["outputs"][branch])


__all__ = [
    "ALLOWED_DEVICES", "ARMS", "DEFAULT_CODE_LOCK", "DEFAULT_SHA", "DEFAULT_SPEC", "HOSTS",
    "PUSHT_TASKS", "REACHER_TASKS", "ROOT", "SEEDS",
    "EvidenceAgeSpecError", "host_tasks", "load_locked_spec", "output_root",
    "resolve_path", "sha256_file", "validate_device", "validate_spec",
]
