#!/usr/bin/env python3
"""Fail-closed protocol loader for the matched-host Wave-1 audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "configs/paper_a_matched_host_v1.yaml"
DEFAULT_SHA = ROOT / "configs/paper_a_matched_host_v1.sha256"
DEFAULT_LOCK = ROOT / "configs/paper_a_matched_host_v1.lock.json"
HOSTS = ("reacher", "pusht", "tworoom")
TARGETS = ("color", "location")
AGES = (4, 8, 15)
ARMS = ("none", "gru", "lstm", "ssm", "fixed_trust")
SEEDS = (0, 1, 2, 3, 4)
ALLOWED_DEVICES = ("cuda:0",)


class MatchedHostSpecError(ValueError):
    """The preregistered matched-host protocol changed or is malformed."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise MatchedHostSpecError("path must be a non-empty repository-relative string")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise MatchedHostSpecError(f"path leaves repository: {value!r}")
    result = (ROOT / relative).resolve()
    try:
        result.relative_to(ROOT.resolve())
    except ValueError as error:
        raise MatchedHostSpecError(f"path leaves repository: {value!r}") from error
    return result


def resolve_input_path(record: Mapping[str, Any]) -> Path:
    """Resolve a pinned input, admitting only the registered tmpfs namespace."""

    value = record.get("path")
    if record.get("external_tmpfs") is True:
        path = Path(value) if isinstance(value, str) else Path()
        allowed = Path("/dev/shm/paper_a_matched_host_v1").resolve()
        result = path.resolve()
        if not path.is_absolute() or allowed not in result.parents:
            raise MatchedHostSpecError(
                "external_tmpfs input must live under the Wave-1 tmpfs namespace")
        return result
    if record.get("external_tmpfs") not in (None, False):
        raise MatchedHostSpecError("external_tmpfs must be boolean when present")
    return resolve_path(value)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise MatchedHostSpecError(f"{label} must be a mapping")
    return value


def _exact(value: Any, expected: tuple[Any, ...], label: str) -> None:
    if value != list(expected):
        raise MatchedHostSpecError(
            f"{label} must be exactly {list(expected)!r}, got {value!r}")


def _identity(record: Any, label: str, *, verify_hash: bool) -> None:
    value = _mapping(record, label)
    path = resolve_input_path(value)
    size = value.get("size")
    digest = value.get("sha256")
    if not path.is_file() or path.stat().st_size != size:
        raise MatchedHostSpecError(f"{label} file identity mismatch: {path}")
    if not isinstance(digest, str) or len(digest) != 64:
        raise MatchedHostSpecError(f"{label}.sha256 is malformed")
    if verify_hash and sha256_file(path) != digest:
        raise MatchedHostSpecError(f"{label} SHA-256 mismatch: {path}")


def validate_spec(spec: Mapping[str, Any], *, verify_inputs: bool = False) -> None:
    if spec.get("schema_version") != 1 \
            or spec.get("study") != "paper-a-matched-host-v1" \
            or spec.get("protocol_status") != "locked-before-formal-outcomes" \
            or spec.get("implementation_lock") \
            != "configs/paper_a_matched_host_v1.lock.json":
        raise MatchedHostSpecError("unexpected matched-host study identity")
    _exact(spec.get("hosts"), HOSTS, "hosts")
    _exact(spec.get("targets"), TARGETS, "targets")
    _exact(spec.get("ages"), AGES, "ages")
    _exact(spec.get("arms"), ARMS, "arms")
    _exact(spec.get("seeds"), SEEDS, "seeds")

    execution = _mapping(spec.get("execution"), "execution")
    _exact(execution.get("allowed_devices"), ALLOWED_DEVICES,
           "execution.allowed_devices")
    if execution.get("default_device") != "cuda:0" \
            or execution.get("explicit_execute_required") is not True \
            or execution.get("deterministic_algorithms") is not True \
            or execution.get("forbid_cuda3") is not True:
        raise MatchedHostSpecError("execution contract changed")

    sequence = _mapping(spec.get("sequence"), "sequence")
    expected_sequence = {
        "num_frames": 20, "decision_index": 19, "cue_length": 3,
        "endpoint_history": 3, "current_observation_excluded": True,
        "frame_skip": 5,
    }
    if any(sequence.get(key) != value
           for key, value in expected_sequence.items()):
        raise MatchedHostSpecError("matched sequence contract changed")
    if sequence.get("predictor_history") != 3 \
            or sequence.get("cue_intervals") != {
                "age-4": [12, 15], "age-8": [8, 11], "age-15": [1, 4]} \
            or sequence.get("endpoint_feature") \
            != "concat(z[16],z[17],z[18],prior_read[19])":
        raise MatchedHostSpecError("matched endpoint or cue interval changed")

    selection = _mapping(spec.get("selection"), "selection")
    if selection.get("train_episodes") != 1200 \
            or selection.get("validation_episodes") != 480 \
            or selection.get("one_sequence_per_episode") is not True \
            or selection.get("episode_disjoint_splits") is not True \
            or selection.get("train_label_seed") != 20260731 \
            or selection.get("validation_label_seed") != 20260732:
        raise MatchedHostSpecError("selection contract changed")

    cue = _mapping(spec.get("cue"), "cue")
    if cue.get("colors") != 4 or cue.get("locations") != 4 \
            or cue.get("all_16_combinations_exactly_balanced") is not True \
            or cue.get("same_rendered_episode_for_both_targets") is not True \
            or float(cue.get("square_side_fraction", -1)) != 0.20:
        raise MatchedHostSpecError("joint cue contract changed")
    if cue.get("rgb_palette") != [
            [230, 57, 70], [40, 160, 84], [45, 108, 223], [239, 174, 45]] \
            or cue.get("normalized_centers") != [
                [0.18, 0.18], [0.82, 0.18], [0.18, 0.82], [0.82, 0.82]]:
        raise MatchedHostSpecError("joint cue palette or centers changed")

    admission = _mapping(spec.get("admission"), "admission")
    if float(admission.get("cue_balanced_accuracy_min", -1)) != 0.75 \
            or float(admission.get("cue_min_class_recall_min", -1)) != 0.70 \
            or float(admission.get("shortcut_margin_above_chance", -1)) != 0.05 \
            or admission.get("all_hosts_targets_ages_must_pass") is not True:
        raise MatchedHostSpecError("admission contract changed")

    training = _mapping(spec.get("carrier_training"), "carrier_training")
    expected_training = {
        "epochs": 100, "batch_size": 64, "learning_rate": 0.0003,
        "weight_decay": 0.00001, "windows_per_episode": 8,
        "age_balanced_mixture": True, "frozen_encoder": True,
        "frozen_predictor": True,
    }
    if any(training.get(key) != value
           for key, value in expected_training.items()):
        raise MatchedHostSpecError("carrier training contract changed")
    if training.get("scheduler") != "cosine-annealing" \
            or training.get("gradient_clip_norm") != 1.0 \
            or training.get("training_rng_offset") != 571000:
        raise MatchedHostSpecError("carrier optimizer auxiliaries changed")

    readout = _mapping(spec.get("readout"), "readout")
    if readout.get("model") != "StandardScaler+LogisticRegression" \
            or readout.get("logistic_c") != 1.0 \
            or readout.get("solver") != "lbfgs" \
            or readout.get("max_iter") != 4000 \
            or readout.get("random_state") != 0:
        raise MatchedHostSpecError("readout contract changed")

    statistics = _mapping(spec.get("statistics"), "statistics")
    if statistics.get("bootstrap_draws") != 20000 \
            or statistics.get("bootstrap_seed") != 20260721 \
            or statistics.get("confidence") != 0.95 \
            or statistics.get("interval") != "percentile" \
            or statistics.get("seed_resampling") \
            != "jointly resample carrier seed indices across hosts" \
            or statistics.get("episode_resampling") \
            != "independently by host and stratified by 16-way joint label" \
            or statistics.get("preserve_pairing") \
            != "targets, ages, and arms within host" \
            or float(statistics.get("equivalence_margin", -1)) != 0.05 \
            or statistics.get("primary_interaction") \
            != ("age-15 PushT-minus-Reacher difference in "
                "fixed-trust-minus-SSM, equally averaged over color and location") \
            or statistics.get("no_pooled_host_memory_score") is not True:
        raise MatchedHostSpecError("statistics contract changed")

    action = _mapping(spec.get("action_normalization"),
                      "action_normalization")
    if action.get("source") \
            != "all finite raw two-dimensional training-source actions per host" \
            or action.get("ddof") != 1 \
            or action.get("order") \
            != ("normalize each raw 2-D action then time-major flatten five "
                "actions to 10-D"):
        raise MatchedHostSpecError("action-normalization contract changed")

    use = _mapping(spec.get("tworoom_use"), "tworoom_use")
    expected_use = {
        "enabled_only_if_all_matched_admissions_pass": True,
        "target": "location", "heldout_episodes": 480,
        "reset_seed": 20260781, "physics_seed": 0,
        "label_seed": 20260782,
        "cue_age": 15, "cue_length": 3, "prefix_model_frames": 20,
        "raw_steps_per_model_frame": 5, "initial_x": 40.0,
        "max_execution_steps": 64, "success_radius": 16.0,
        "oracle_success_min": 0.90,
        "oracle_per_class_success_min": 0.90,
        "off_diagonal_false_success_max": 0.05,
        "replay_fidelity_min": 0.99,
        "realized_random_goal_baseline": True,
        "fixed_physics_across_arms_and_episodes": True,
    }
    if any(use.get(key) != value for key, value in expected_use.items()) \
            or use.get("shared_prefix_target") != [128.0, 49.0] \
            or use.get("initial_y_range") != [28.0, 196.0] \
            or use.get("goal_waypoints") != [
                [154.0, 40.0], [196.0, 40.0],
                [154.0, 174.0], [196.0, 174.0]] \
            or use.get("consumer") \
            != "arm-blind StandardScaler+multinomial LogisticRegression" \
            or use.get("controller") \
            != "pinned deterministic TwoRoom ExpertPolicy" \
            or use.get("claim_boundary") \
            != ("external memory-conditioned navigation execution, not native "
                "LeWM planning"):
        raise MatchedHostSpecError("TwoRoom downstream-use contract changed")
    for key, expected_hash in (
            ("upstream_environment",
             "5e1d392de5b02472062dbe872aded67fd465fcc8f7eaa1c02a753b2fc31c61f0"),
            ("upstream_controller",
             "5939318d2a671ce00abf46be74ae22cf2452ea48426eb9dc651d04514849e4f4")):
        source = _mapping(use.get(key), f"tworoom_use.{key}")
        path = resolve_path(source.get("path"))
        if not path.is_file() or source.get("sha256") != expected_hash \
                or sha256_file(path) != expected_hash:
            raise MatchedHostSpecError(f"TwoRoom pinned source changed: {key}")
    if use["upstream_environment"].get("revision") \
            != "0ef3856875e70a1283e637fcd2ab936eae6c4e6f":
        raise MatchedHostSpecError("TwoRoom upstream revision changed")

    inputs = _mapping(spec.get("inputs"), "inputs")
    expected_identities = {
        "reacher": ["weights"],
        "pusht": ["config", "weights", "dataset"],
        "tworoom": ["config", "weights", "archive", "dataset"],
    }
    expected_sources = {
        "reacher": (
            "quentinll/lewm-reacher",
            "62adae4b71dc474ddf8f794c476ebfe737a743ca", None),
        "pusht": (
            "quentinll/lewm-pusht",
            "22b330c28c27ead4bfd1888615af1340e3fe9052",
            "655cd446b9929369d7d406001da85c15d1457850"),
        "tworoom": (
            "quentinll/lewm-tworooms",
            "77adaae0bc31deab21c93740d1f8bb947cd0bdec",
            "6903a2de048b13819d812da0b4dd661290bc01e4"),
    }
    for host in HOSTS:
        record = _mapping(inputs.get(host), f"inputs.{host}")
        repo, model_revision, dataset_revision = expected_sources[host]
        if record.get("repo_id") != repo \
                or record.get("model_revision") != model_revision \
                or (dataset_revision is not None
                    and record.get("dataset_revision") != dataset_revision):
            raise MatchedHostSpecError(f"inputs.{host} source revision changed")
        if record.get("identity_records") != expected_identities[host]:
            raise MatchedHostSpecError(
                f"inputs.{host}.identity_records changed")
        for key in record.get("identity_records", ()):
            _identity(record.get(key), f"inputs.{host}.{key}",
                      verify_hash=verify_inputs)
    if inputs["reacher"].get("kind") != "fresh-dm-control-reacher-easy" \
            or inputs["reacher"].get("raw_frame_size") != 64 \
            or inputs["reacher"].get("target_alpha_zero") is not True \
            or inputs["pusht"].get("kind") != "official-root-hdf5" \
            or inputs["pusht"].get("state_key") != "state" \
            or inputs["tworoom"].get("kind") != "official-root-hdf5" \
            or inputs["tworoom"].get("state_key") != "observation":
        raise MatchedHostSpecError("host input adapter contract changed")
    if inputs["reacher"].get("train_base_seed") != 20260741 \
            or inputs["reacher"].get("validation_base_seed") != 20260742 \
            or inputs["pusht"].get("split_seed") != 20260751 \
            or inputs["pusht"].get("start_seed") != 20260752 \
            or inputs["tworoom"].get("split_seed") != 20260761 \
            or inputs["tworoom"].get("start_seed") != 20260762:
        raise MatchedHostSpecError("host selection seeds changed")
    if inputs["tworoom"].get("release_paper_history_discrepancy") != {
            "authenticated_config_history": 3,
            "paper_appendix_history": 1,
            "executable_truth": "authenticated released config and weights",
            }:
        raise MatchedHostSpecError("TwoRoom release discrepancy record changed")

    outputs = _mapping(spec.get("outputs"), "outputs")
    root = resolve_path(outputs.get("root"))
    for key in ("cache", "carriers", "logs", "use"):
        child = resolve_path(outputs.get(key))
        if root not in child.parents:
            raise MatchedHostSpecError(f"outputs.{key} is outside output root")


def _verify_sidecar(spec_path: Path, sha_path: Path) -> str:
    if not sha_path.is_file():
        raise MatchedHostSpecError(f"missing protocol SHA sidecar: {sha_path}")
    fields = sha_path.read_text().strip().split()
    digest = sha256_file(spec_path)
    if len(fields) != 2 or fields[0] != digest \
            or fields[1] != spec_path.name:
        raise MatchedHostSpecError("protocol SHA sidecar mismatch")
    return digest


def _verify_lock(spec: Mapping[str, Any], digest: str) -> dict[str, Any]:
    path = resolve_path(spec.get("implementation_lock"))
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise MatchedHostSpecError(f"cannot read implementation lock: {error}") from error
    if value.get("schema_version") != 1 \
            or value.get("study") != spec.get("study") \
            or value.get("spec_sha256") != digest \
            or value.get("locked_before_formal_outcomes") is not True:
        raise MatchedHostSpecError("implementation lock identity mismatch")
    producers = value.get("producers")
    if not isinstance(producers, dict) or not producers:
        raise MatchedHostSpecError("implementation lock has no producers")
    for relative, expected in producers.items():
        source = resolve_path(relative)
        if not source.is_file() or sha256_file(source) != expected:
            raise MatchedHostSpecError(f"locked producer changed: {relative}")
    return {"path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path), "producers": len(producers)}


def load_locked_spec(spec_path: Path = DEFAULT_SPEC,
                     sha_path: Path = DEFAULT_SHA, *,
                     verify_inputs: bool = True) -> dict[str, Any]:
    spec_path, sha_path = spec_path.resolve(), sha_path.resolve()
    digest = _verify_sidecar(spec_path, sha_path)
    value = yaml.safe_load(spec_path.read_text())
    if not isinstance(value, dict):
        raise MatchedHostSpecError("protocol YAML must contain one mapping")
    validate_spec(value, verify_inputs=verify_inputs)
    result = dict(value)
    result["_lock"] = {
        "path": str(spec_path.relative_to(ROOT)), "sha256": digest,
        "sidecar": str(sha_path.relative_to(ROOT)),
        "sidecar_sha256": sha256_file(sha_path),
        "implementation": _verify_lock(result, digest),
    }
    return result


def validate_device(spec: Mapping[str, Any], device: str) -> str:
    if device not in spec["execution"]["allowed_devices"]:
        raise MatchedHostSpecError(
            f"device {device!r} forbidden; only physical cuda:0 is admitted")
    return device


def output_path(spec: Mapping[str, Any], key: str) -> Path:
    if key not in ("root", "cache", "carriers", "logs", "use"):
        raise MatchedHostSpecError(f"unknown output key {key!r}")
    return resolve_path(spec["outputs"][key])


__all__ = [
    "AGES", "ALLOWED_DEVICES", "ARMS", "DEFAULT_LOCK", "DEFAULT_SHA",
    "DEFAULT_SPEC", "HOSTS", "MatchedHostSpecError", "ROOT", "SEEDS",
    "TARGETS", "load_locked_spec", "output_path", "resolve_path",
    "resolve_input_path", "sha256_file", "validate_device", "validate_spec",
]
