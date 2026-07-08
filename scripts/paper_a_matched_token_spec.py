#!/usr/bin/env python3
"""Fail-closed protocol loader for adaptive matched composite-token audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "configs/paper_a_matched_token_v1.yaml"
DEFAULT_SHA = ROOT / "configs/paper_a_matched_token_v1.sha256"
DEFAULT_LOCK = ROOT / "configs/paper_a_matched_token_v1.lock.json"
HOSTS = ("reacher", "pusht", "tworoom")
HDF_HOSTS = ("pusht", "tworoom")
AGES = (4, 8, 15)
ARMS = ("none", "gru", "lstm", "ssm", "fixed_trust")
SEEDS = (0, 1, 2, 3, 4)


class MatchedTokenSpecError(ValueError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise MatchedTokenSpecError("path must be a repository-relative string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise MatchedTokenSpecError(f"path leaves repository: {value}")
    result = (ROOT / path).resolve()
    try:
        result.relative_to(ROOT.resolve())
    except ValueError as error:
        raise MatchedTokenSpecError(f"path leaves repository: {value}") from error
    return result


def resolve_input_path(record: Mapping[str, Any]) -> Path:
    if record.get("external_tmpfs") is True:
        path = Path(record.get("path", ""))
        root = Path("/dev/shm/paper_a_matched_host_v1").resolve()
        result = path.resolve()
        if not path.is_absolute() or root not in result.parents:
            raise MatchedTokenSpecError("tmpfs input leaves pinned namespace")
        return result
    return resolve_path(record.get("path"))


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise MatchedTokenSpecError(f"{name} must be a mapping")
    return value


def _identity(record: Any, name: str, verify: bool) -> None:
    value = _mapping(record, name)
    path = resolve_input_path(value)
    if not path.is_file() or path.stat().st_size != value.get("size"):
        raise MatchedTokenSpecError(f"{name} size differs")
    digest = value.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64 \
            or (verify and sha256_file(path) != digest):
        raise MatchedTokenSpecError(f"{name} hash differs")


def validate_spec(spec: Mapping[str, Any], *, verify_inputs: bool = False) -> None:
    if spec.get("schema_version") != 1 \
            or spec.get("study") != "paper-a-matched-token-v1" \
            or spec.get("protocol_status") \
            != "adaptive-locked-before-formal-outcomes" \
            or spec.get("implementation_lock") \
            != "configs/paper_a_matched_token_v1.lock.json" \
            or spec.get("hosts") != list(HOSTS) \
            or spec.get("target") != "token" \
            or spec.get("nuisance") != "location" \
            or spec.get("ages") != list(AGES) \
            or spec.get("arms") != list(ARMS) \
            or spec.get("seeds") != list(SEEDS):
        raise MatchedTokenSpecError("study identity changed")
    origin = _mapping(spec.get("adaptive_origin"), "adaptive_origin")
    expected_origin = {
        "parent_study": "paper-a-matched-host-v1",
        "trigger": ("the common simple color-location cue failed preregistered "
                    "availability on at least one target or age in every host"),
        "adaptation": ("replace simple color identity with a larger bordered "
                       "composite color-plus-geometry token; retain balanced "
                       "location nuisance"),
        "timing": ("declared after all V1 admission receipts and before every "
                   "matched-token outcome"),
        "v1_admission_metrics_used_for_adaptation": True,
        "v1_admission_metrics_used_for_matched_token_inference": False,
        "v1_carrier_outcomes_used": False, "preserve_v1_failure": True,
        "unsealed_color_draft_used": False,
    }
    if any(origin.get(key) != value for key, value in expected_origin.items()):
        raise MatchedTokenSpecError("adaptive origin changed")
    for key in ("v1_spec", "v1_implementation_lock"):
        _identity(origin.get(key), f"adaptive_origin.{key}", True)
    receipts = _mapping(origin.get("v1_host_receipts"), "V1 receipts")
    failed_hosts = set()
    for host in HOSTS:
        record = _mapping(receipts.get(host), f"V1 receipt {host}")
        _identity(record, f"V1 receipt {host}", True)
        value = json.loads(resolve_path(record["path"]).read_text())
        if value.get("study") != "paper-a-matched-host-v1" \
                or value.get("host") != host \
                or value.get("status") != "stopped-admission-failure":
            raise MatchedTokenSpecError(f"V1 receipt identity differs: {host}")
        if any(not target.get("admitted", False)
               for age in value.get("admission", {}).values()
               for target in age.values()):
            failed_hosts.add(host)
    if failed_hosts != set(HOSTS):
        raise MatchedTokenSpecError("V1 does not support three-host adaptation")
    execution = _mapping(spec.get("execution"), "execution")
    if execution.get("allowed_devices") != ["cuda:0"] \
            or execution.get("default_device") != "cuda:0" \
            or execution.get("explicit_execute_required") is not True \
            or execution.get("deterministic_algorithms") is not True \
            or execution.get("forbid_cuda3") is not True:
        raise MatchedTokenSpecError("execution contract changed")
    inputs = _mapping(spec.get("inputs"), "inputs")
    identities = {
        "reacher": ("weights",),
        "pusht": ("config", "weights", "dataset"),
        "tworoom": ("config", "weights", "archive", "dataset"),
    }
    seeds = {
        "reacher": (20260941, 20260942),
        "pusht": (20260951, 20260952),
        "tworoom": (20260961, 20260962),
    }
    for host in HOSTS:
        record = _mapping(inputs.get(host), f"inputs.{host}")
        if tuple(record.get("identity_records", ())) != identities[host]:
            raise MatchedTokenSpecError(f"input identity list changed: {host}")
        for key in identities[host]:
            _identity(record.get(key), f"inputs.{host}.{key}", verify_inputs)
        actual_seeds = ((record.get("train_base_seed"),
                         record.get("validation_base_seed"))
                        if host == "reacher"
                        else (record.get("split_seed"), record.get("start_seed")))
        if actual_seeds != seeds[host]:
            raise MatchedTokenSpecError(f"fresh seeds changed: {host}")
    sequence = _mapping(spec.get("sequence"), "sequence")
    if sequence != {
            "num_frames": 20, "frame_skip": 5, "decision_index": 19,
            "cue_length": 3,
            "cue_intervals": {"age-4": [12, 15], "age-8": [8, 11],
                              "age-15": [1, 4]},
            "predictor_history": 3, "endpoint_history": 3,
            "endpoint_feature": "concat(z[16],z[17],z[18],prior_read[19])",
            "current_observation_excluded": True}:
        raise MatchedTokenSpecError("sequence contract changed")
    selection = _mapping(spec.get("selection"), "selection")
    if selection != {
            "train_episodes": 1200, "validation_episodes": 480,
            "train_label_seed": 20260931,
            "validation_label_seed": 20260932,
            "one_sequence_per_episode": True,
            "episode_disjoint_splits": True,
            "exclude_v1_hdf_episode_indices_before_permutation": True}:
        raise MatchedTokenSpecError("selection contract changed")
    token = _mapping(spec.get("token"), "token")
    if token.get("classes") != ["vertical-bar", "horizontal-bar", "x", "plus"] \
            or token.get("rgb_palette") != [
                [225, 35, 45], [25, 155, 65], [35, 90, 225], [235, 155, 20]] \
            or token.get("normalized_centers") != [
                [0.20, 0.20], [0.80, 0.20], [0.20, 0.80], [0.80, 0.80]] \
            or token.get("square_side_fraction") != 0.28 \
            or token.get("geometry_thickness_fraction") != 0.22 \
            or token.get("all_16_combinations_exactly_balanced") is not True \
            or token.get("location_role") \
            != "exact-balanced randomized nuisance":
        raise MatchedTokenSpecError("composite token contract changed")
    admission = _mapping(spec.get("admission"), "admission")
    if admission != {
            "chance": 0.25, "cue_balanced_accuracy_min": 0.75,
            "cue_min_class_recall_min": 0.70, "shortcut_ceiling": 0.30,
            "all_hosts_ages_must_pass": True,
            "no_carrier_training_if_any_gate_fails": True,
            "frozen_host_hash_required": True}:
        raise MatchedTokenSpecError("admission contract changed")
    training = _mapping(spec.get("carrier_training"), "carrier_training")
    expected_training = {
        "epochs": 100, "batch_size": 64, "learning_rate": 0.0003,
        "weight_decay": 0.00001, "scheduler": "cosine-annealing",
        "gradient_clip_norm": 1.0, "windows_per_episode": 8,
        "training_rng_offset": 591000, "age_balanced_mixture": True,
        "frozen_encoder": True, "frozen_predictor": True,
    }
    if training != expected_training:
        raise MatchedTokenSpecError("training contract changed")
    use = _mapping(spec.get("tworoom_use"), "tworoom_use")
    required_use = {
        "enabled_only_if_all_token_admissions_pass": True,
        "target": "token", "nuisance": "location",
        "heldout_episodes": 480, "split_seed": 20260971,
        "start_seed": 20260972, "label_seed": 20260973,
        "random_goal_seed": 20260974,
        "exclude_v1_and_matched_token_train_validation_episodes": True,
        "cue_age": 15, "state_position_indices": [0, 1],
        "token_to_goal_index": [0, 1, 2, 3], "physics_seed": 0,
        "max_execution_steps": 64, "success_radius": 16.0,
        "oracle_success_min": 0.90, "oracle_per_class_success_min": 0.90,
        "off_diagonal_false_success_max": 0.05,
        "replay_fidelity_min": 0.99,
        "bootstrap_draws": 20000, "bootstrap_seed": 20260975,
    }
    if any(use.get(key) != value for key, value in required_use.items()) \
            or use.get("goal_waypoints") != [
                [154.0, 40.0], [196.0, 40.0],
                [154.0, 174.0], [196.0, 174.0]]:
        raise MatchedTokenSpecError("TwoRoom use contract changed")
    for key, digest in (
            ("upstream_environment",
             "5e1d392de5b02472062dbe872aded67fd465fcc8f7eaa1c02a753b2fc31c61f0"),
            ("upstream_controller",
             "5939318d2a671ce00abf46be74ae22cf2452ea48426eb9dc651d04514849e4f4")):
        record = _mapping(use.get(key), f"tworoom_use.{key}")
        path = resolve_path(record.get("path"))
        if record.get("sha256") != digest or sha256_file(path) != digest:
            raise MatchedTokenSpecError(f"use source changed: {key}")
    outputs = _mapping(spec.get("outputs"), "outputs")
    root = resolve_path(outputs.get("root"))
    for key in ("cache", "carriers", "logs", "use"):
        if root not in resolve_path(outputs.get(key)).parents:
            raise MatchedTokenSpecError(f"output leaves root: {key}")


def _verify_lock(spec: Mapping[str, Any], spec_hash: str) -> dict[str, Any]:
    path = resolve_path(spec["implementation_lock"])
    value = json.loads(path.read_text())
    if value.get("study") != spec["study"] \
            or value.get("spec_sha256") != spec_hash \
            or value.get("locked_before_matched_token_outcomes") is not True:
        raise MatchedTokenSpecError("implementation lock differs")
    for relative, digest in _mapping(
            value.get("producers"), "lock.producers").items():
        source = resolve_path(relative)
        if not source.is_file() or sha256_file(source) != digest:
            raise MatchedTokenSpecError(f"locked producer changed: {relative}")
    exclusions = _mapping(value.get("v1_hdf_exclusions"), "lock.exclusions")
    for host in HDF_HOSTS:
        record = _mapping(exclusions.get(host), f"exclusion {host}")
        indices = record.get("episode_indices")
        if not isinstance(indices, list) or len(indices) != 1680 \
                or indices != sorted(set(indices)) \
                or record.get("indices_sha256") != hashlib.sha256(
                    json.dumps(indices, separators=(",", ":")).encode()
                ).hexdigest():
            raise MatchedTokenSpecError(f"locked exclusion differs: {host}")
        for artifact in record.get("artifacts", []):
            source = resolve_path(artifact.get("path"))
            if not source.is_file() or source.stat().st_size != artifact.get("size") \
                    or sha256_file(source) != artifact.get("sha256"):
                raise MatchedTokenSpecError("V1 exclusion artifact changed")
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path),
            "v1_hdf_exclusions": exclusions,
            "producers": len(value["producers"])}


def load_locked_spec(spec_path: Path = DEFAULT_SPEC,
                     sha_path: Path = DEFAULT_SHA, *,
                     verify_inputs: bool = True) -> dict[str, Any]:
    fields = sha_path.read_text().strip().split()
    digest = sha256_file(spec_path)
    if len(fields) != 2 or fields != [digest, spec_path.name]:
        raise MatchedTokenSpecError("spec sidecar differs")
    value = yaml.safe_load(spec_path.read_text())
    validate_spec(value, verify_inputs=verify_inputs)
    result = dict(value)
    result["_lock"] = {
        "path": str(spec_path.relative_to(ROOT)), "sha256": digest,
        "sidecar": str(sha_path.relative_to(ROOT)),
        "sidecar_sha256": sha256_file(sha_path),
        "implementation": _verify_lock(result, digest),
    }
    return result


def v1_excluded_episode_indices(spec: Mapping[str, Any], host: str
                                ) -> tuple[int, ...]:
    if host not in HDF_HOSTS:
        return ()
    return tuple(spec["_lock"]["implementation"][
        "v1_hdf_exclusions"][host]["episode_indices"])


def validate_device(spec: Mapping[str, Any], device: str) -> str:
    if device not in spec["execution"]["allowed_devices"]:
        raise MatchedTokenSpecError("only physical cuda:0 is allowed")
    return device


def output_path(spec: Mapping[str, Any], key: str) -> Path:
    if key not in ("root", "cache", "carriers", "logs", "use"):
        raise MatchedTokenSpecError(f"unknown output {key}")
    return resolve_path(spec["outputs"][key])


__all__ = [
    "AGES", "ARMS", "DEFAULT_LOCK", "DEFAULT_SHA", "DEFAULT_SPEC",
    "HDF_HOSTS", "HOSTS", "ROOT", "SEEDS", "load_locked_spec",
    "output_path", "resolve_input_path", "resolve_path", "sha256_file",
    "v1_excluded_episode_indices", "validate_device", "validate_spec",
]
