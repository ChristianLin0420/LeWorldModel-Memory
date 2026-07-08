#!/usr/bin/env python3
"""Fail-closed loader for the adaptive Wave-1b color-only protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "configs/paper_a_matched_color_v1.yaml"
DEFAULT_SHA = ROOT / "configs/paper_a_matched_color_v1.sha256"
DEFAULT_LOCK = ROOT / "configs/paper_a_matched_color_v1.lock.json"
HOSTS = ("reacher", "pusht", "tworoom")
TARGETS = ("color",)
AGES = (4, 8, 15)
ARMS = ("none", "gru", "lstm", "ssm", "fixed_trust")
SEEDS = (0, 1, 2, 3, 4)
ALLOWED_DEVICES = ("cuda:0",)
HDF_HOSTS = ("pusht", "tworoom")


class MatchedColorSpecError(ValueError):
    """The adaptive matched-color protocol changed or is malformed."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _indices_sha256(indices: list[int]) -> str:
    payload = json.dumps(indices, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def resolve_path(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise MatchedColorSpecError(
            "path must be a non-empty repository-relative string")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise MatchedColorSpecError(f"path leaves repository: {value!r}")
    result = (ROOT / relative).resolve()
    try:
        result.relative_to(ROOT.resolve())
    except ValueError as error:
        raise MatchedColorSpecError(f"path leaves repository: {value!r}") from error
    return result


def resolve_input_path(record: Mapping[str, Any]) -> Path:
    value = record.get("path")
    if record.get("external_tmpfs") is True:
        path = Path(value) if isinstance(value, str) else Path()
        allowed = Path("/dev/shm/paper_a_matched_host_v1").resolve()
        result = path.resolve()
        if not path.is_absolute() or allowed not in result.parents:
            raise MatchedColorSpecError(
                "external tmpfs input is outside the pinned Wave-1 namespace")
        return result
    if record.get("external_tmpfs") not in (None, False):
        raise MatchedColorSpecError("external_tmpfs must be boolean")
    return resolve_path(value)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise MatchedColorSpecError(f"{label} must be a mapping")
    return value


def _exact(value: Any, expected: tuple[Any, ...], label: str) -> None:
    if value != list(expected):
        raise MatchedColorSpecError(
            f"{label} must be exactly {list(expected)!r}, got {value!r}")


def _identity(record: Any, label: str, *, verify_hash: bool) -> None:
    value = _mapping(record, label)
    path = resolve_input_path(value)
    size, digest = value.get("size"), value.get("sha256")
    if not path.is_file() or path.stat().st_size != size:
        raise MatchedColorSpecError(f"{label} file identity mismatch: {path}")
    if not isinstance(digest, str) or len(digest) != 64:
        raise MatchedColorSpecError(f"{label}.sha256 is malformed")
    if verify_hash and sha256_file(path) != digest:
        raise MatchedColorSpecError(f"{label} SHA-256 mismatch: {path}")


def _validate_adaptive_origin(spec: Mapping[str, Any]) -> None:
    value = _mapping(spec.get("adaptive_origin"), "adaptive_origin")
    expected = {
        "parent_study": "paper-a-matched-host-v1",
        "adaptation": (
            "drop failed location target; retain predeclared color target unchanged"),
        "trigger": (
            "V1 Reacher location cue-availability gate failed while color passed "
            "at all ages"),
        "timing": (
            "declared after V1 availability stop and before every Wave-1b formal "
            "outcome"),
        "v1_carrier_outcomes_used": False,
        "v1_admission_metrics_used_for_adaptation": True,
        "v1_admission_metrics_used_for_wave1b_inference": False,
        "preserve_v1_failure": True,
        "hdf_v1_exclusion_policy": (
            "union episode_index from all four identity-verified V1 HDF train "
            "and validation base caches pinned with sidecars and host receipts "
            "at Wave-1b seal time"),
    }
    if any(value.get(key) != expected_value
           for key, expected_value in expected.items()):
        raise MatchedColorSpecError("adaptive-origin contract changed")
    identities = {
        "v1_spec": (7649,
                    "5febf1c31d8a9f73c83a4d26ba4ed0f9934a23e6099840e50ceba32b4b740b7f"),
        "v1_spec_sidecar": (95,
                            "73a653b76a92ed016d075c4cde7ac57cc43c9d3fae28cbabb14c7a23023eade3"),
        "v1_implementation_lock": (3461,
                                   "83c2ae48ea53e2080606fa7e15e30339668deb338fd556527025e9418b076998"),
        "v1_stop_receipt": (22068,
                            "e7d445bbdeb7559ee7af6b3476783339f6f4c344165d8948caae89d26a20eedd"),
    }
    for key, (size, digest) in identities.items():
        record = _mapping(value.get(key), f"adaptive_origin.{key}")
        if record.get("size") != size or record.get("sha256") != digest:
            raise MatchedColorSpecError(f"adaptive identity changed: {key}")
        _identity(record, f"adaptive_origin.{key}", verify_hash=True)
    receipt = json.loads(resolve_path(value["v1_stop_receipt"]["path"]).read_text())
    if receipt.get("study") != "paper-a-matched-host-v1" \
            or receipt.get("host") != "reacher" \
            or receipt.get("status") != "stopped-admission-failure" \
            or receipt.get("all_targets_ages_admitted") is not False:
        raise MatchedColorSpecError("V1 adaptive stop receipt has wrong identity")
    admission = _mapping(receipt.get("admission"), "V1 receipt admission")
    color_pass = all(
        _mapping(admission.get(f"age-{age}"), f"V1 age-{age}")
        .get("color", {}).get("admitted") is True for age in AGES)
    location_failed = any(
        _mapping(admission.get(f"age-{age}"), f"V1 age-{age}")
        .get("location", {}).get("admitted") is False for age in AGES)
    if not color_pass or not location_failed:
        raise MatchedColorSpecError("V1 receipt does not support declared adaptation")


def validate_spec(spec: Mapping[str, Any], *, verify_inputs: bool = False) -> None:
    if spec.get("schema_version") != 1 \
            or spec.get("study") != "paper-a-matched-color-v1" \
            or spec.get("protocol_status") \
            != "adaptive-locked-before-formal-outcomes" \
            or spec.get("implementation_lock") \
            != "configs/paper_a_matched_color_v1.lock.json":
        raise MatchedColorSpecError("unexpected Wave-1b study identity")
    _exact(spec.get("hosts"), HOSTS, "hosts")
    _exact(spec.get("targets"), TARGETS, "targets")
    _exact(spec.get("ages"), AGES, "ages")
    _exact(spec.get("arms"), ARMS, "arms")
    _exact(spec.get("seeds"), SEEDS, "seeds")
    _validate_adaptive_origin(spec)

    execution = _mapping(spec.get("execution"), "execution")
    _exact(execution.get("allowed_devices"), ALLOWED_DEVICES,
           "execution.allowed_devices")
    if execution.get("default_device") != "cuda:0" \
            or execution.get("explicit_execute_required") is not True \
            or execution.get("deterministic_algorithms") is not True \
            or execution.get("forbid_cuda3") is not True \
            or execution.get("initial_physical_gpu") != 0:
        raise MatchedColorSpecError("execution contract changed")

    sequence = _mapping(spec.get("sequence"), "sequence")
    expected_sequence = {
        "num_frames": 20, "frame_skip": 5, "decision_index": 19,
        "cue_length": 3, "predictor_history": 3, "endpoint_history": 3,
        "current_observation_excluded": True,
    }
    if any(sequence.get(key) != expected for key, expected in
           expected_sequence.items()) \
            or sequence.get("cue_intervals") != {
                "age-4": [12, 15], "age-8": [8, 11], "age-15": [1, 4]} \
            or sequence.get("endpoint_feature") \
            != "concat(z[16],z[17],z[18],prior_read[19])":
        raise MatchedColorSpecError("sequence or H3 endpoint changed")

    selection = _mapping(spec.get("selection"), "selection")
    if selection.get("train_episodes") != 1200 \
            or selection.get("validation_episodes") != 480 \
            or selection.get("one_sequence_per_episode") is not True \
            or selection.get("episode_disjoint_splits") is not True \
            or selection.get("exclude_v1_hdf_episode_indices_before_permutation") \
            is not True \
            or selection.get("train_label_seed") != 20260831 \
            or selection.get("validation_label_seed") != 20260832:
        raise MatchedColorSpecError("selection contract changed")

    cue = _mapping(spec.get("cue"), "cue")
    if cue.get("colors") != 4 or cue.get("locations") != 4 \
            or cue.get("target") != "color" \
            or cue.get("location_role") \
            != "exact-balanced randomized nuisance" \
            or cue.get("all_16_combinations_exactly_balanced") is not True \
            or cue.get("labels_independent_of_base_episode") is not True \
            or cue.get("offcue_bytes_identical") is not True \
            or float(cue.get("square_side_fraction", -1)) != 0.20 \
            or cue.get("rgb_palette") != [
                [230, 57, 70], [40, 160, 84], [45, 108, 223], [239, 174, 45]] \
            or cue.get("normalized_centers") != [
                [0.18, 0.18], [0.82, 0.18], [0.18, 0.82], [0.82, 0.82]]:
        raise MatchedColorSpecError("color-by-location nuisance cue changed")

    admission = _mapping(spec.get("admission"), "admission")
    if float(admission.get("chance", -1)) != 0.25 \
            or float(admission.get("cue_balanced_accuracy_min", -1)) != 0.75 \
            or float(admission.get("cue_min_class_recall_min", -1)) != 0.70 \
            or float(admission.get("shortcut_ceiling", -1)) != 0.30 \
            or admission.get("all_hosts_ages_must_pass") is not True \
            or admission.get("no_carrier_training_if_any_gate_fails") is not True \
            or admission.get("frozen_host_hash_required") is not True:
        raise MatchedColorSpecError("admission contract changed")
    if admission.get("shortcut_features") != {
            "final_context_latent": "concat(z[16],z[17],z[18])",
            "final_action": "concat(a[15],a[16],a[17],a[18])",
            "final_state": "state[19]"}:
        raise MatchedColorSpecError("shortcut feature contract changed")

    action = _mapping(spec.get("action_normalization"), "action_normalization")
    if action.get("source") \
            != "all finite raw two-dimensional training-source actions per host" \
            or action.get("ddof") != 1 \
            or action.get("order") != (
                "normalize each raw 2-D action then time-major flatten five "
                "actions to 10-D"):
        raise MatchedColorSpecError("action-normalization contract changed")

    training = _mapping(spec.get("carrier_training"), "carrier_training")
    expected_training = {
        "epochs": 100, "batch_size": 64, "learning_rate": 0.0003,
        "weight_decay": 0.00001, "scheduler": "cosine-annealing",
        "gradient_clip_norm": 1.0, "windows_per_episode": 8,
        "training_rng_offset": 581000, "age_balanced_mixture": True,
        "frozen_encoder": True, "frozen_predictor": True,
    }
    if any(training.get(key) != expected for key, expected in
           expected_training.items()):
        raise MatchedColorSpecError("carrier training contract changed")
    readout = _mapping(spec.get("readout"), "readout")
    if readout.get("model") != "StandardScaler+LogisticRegression" \
            or readout.get("logistic_c") != 1.0 \
            or readout.get("solver") != "lbfgs" \
            or readout.get("max_iter") != 4000 \
            or readout.get("random_state") != 0 \
            or readout.get("metric") != "balanced_accuracy" \
            or readout.get("separate_readout_per_age") is not True:
        raise MatchedColorSpecError("readout contract changed")
    statistics = _mapping(spec.get("statistics"), "statistics")
    if statistics.get("bootstrap_draws") != 20000 \
            or statistics.get("bootstrap_seed") != 20260821 \
            or statistics.get("confidence") != 0.95 \
            or statistics.get("interval") != "percentile" \
            or statistics.get("seed_resampling") \
            != "jointly resample carrier seed indices across hosts" \
            or statistics.get("episode_resampling") \
            != "independently by host and stratified by 16-way joint label" \
            or statistics.get("preserve_pairing") \
            != "ages and arms within host" \
            or statistics.get("equivalence_margin") != 0.05 \
            or statistics.get("primary_interaction") \
            != ("age-15 PushT-minus-Reacher difference in fixed-trust-minus-"
                "SSM color balanced accuracy") \
            or statistics.get("nuisance_location_summary") \
            != ("per-location accuracy and worst-location accuracy for every "
                "host-arm-age cell") \
            or statistics.get("no_pooled_host_memory_score") is not True:
        raise MatchedColorSpecError("statistics contract changed")

    use = _mapping(spec.get("tworoom_use"), "tworoom_use")
    expected_use = {
        "enabled_only_if_all_color_admissions_pass": True,
        "target": "color", "nuisance": "location",
        "heldout_episodes": 480, "split_seed": 20260871,
        "start_seed": 20260872, "label_seed": 20260873,
        "random_goal_seed": 20260874,
        "exclude_locked_v1_and_wave1b_train_validation_episodes": True,
        "cue_age": 15, "cue_length": 3, "model_frames": 20,
        "decision_index": 19, "state_position_indices": [0, 1],
        "color_to_goal_index": [0, 1, 2, 3], "physics_seed": 0,
        "fixed_physics_across_arms_episodes_and_counterfactual_goals": True,
        "max_execution_steps": 64, "success_radius": 16.0,
        "oracle_success_min": 0.90,
        "oracle_per_class_success_min": 0.90,
        "off_diagonal_false_success_max": 0.05,
        "replay_fidelity_min": 0.99,
        "bootstrap_draws": 20000, "bootstrap_seed": 20260875,
    }
    if any(use.get(key) != value for key, value in expected_use.items()) \
            or use.get("goal_waypoints") != [
                [154.0, 40.0], [196.0, 40.0],
                [154.0, 174.0], [196.0, 174.0]] \
            or use.get("consumer") != (
                "one shared arm-blind StandardScaler+multinomial "
                "LogisticRegression per seed, equally pooled over all five arms") \
            or use.get("controller") \
            != "pinned deterministic TwoRoom ExpertPolicy" \
            or use.get("claim_boundary") != (
                "external color-memory-conditioned TwoRoom waypoint execution, "
                "not native LeWM planning"):
        raise MatchedColorSpecError("TwoRoom color-use contract changed")
    expected_use_sources = {
        "upstream_environment": (
            "5e1d392de5b02472062dbe872aded67fd465fcc8f7eaa1c02a753b2fc31c61f0",
            "0ef3856875e70a1283e637fcd2ab936eae6c4e6f"),
        "upstream_controller": (
            "5939318d2a671ce00abf46be74ae22cf2452ea48426eb9dc651d04514849e4f4",
            None),
    }
    for key, (digest, revision) in expected_use_sources.items():
        record = _mapping(use.get(key), f"tworoom_use.{key}")
        path = resolve_path(record.get("path"))
        if record.get("sha256") != digest or not path.is_file() \
                or sha256_file(path) != digest \
                or (revision is not None and record.get("revision") != revision):
            raise MatchedColorSpecError(f"TwoRoom use source changed: {key}")

    inputs = _mapping(spec.get("inputs"), "inputs")
    expected_sources = {
        "reacher": ("quentinll/lewm-reacher",
                    "62adae4b71dc474ddf8f794c476ebfe737a743ca", None,
                    ("weights",)),
        "pusht": ("quentinll/lewm-pusht",
                  "22b330c28c27ead4bfd1888615af1340e3fe9052",
                  "655cd446b9929369d7d406001da85c15d1457850",
                  ("config", "weights", "dataset")),
        "tworoom": ("quentinll/lewm-tworooms",
                    "77adaae0bc31deab21c93740d1f8bb947cd0bdec",
                    "6903a2de048b13819d812da0b4dd661290bc01e4",
                    ("config", "weights", "archive", "dataset")),
    }
    for host in HOSTS:
        record = _mapping(inputs.get(host), f"inputs.{host}")
        repo, revision, dataset_revision, identities = expected_sources[host]
        if record.get("repo_id") != repo \
                or record.get("model_revision") != revision \
                or (dataset_revision is not None and
                    record.get("dataset_revision") != dataset_revision):
            raise MatchedColorSpecError(f"inputs.{host} source changed")
        _exact(record.get("identity_records"), identities,
               f"inputs.{host}.identity_records")
        for key in identities:
            _identity(record.get(key), f"inputs.{host}.{key}",
                      verify_hash=verify_inputs)
    if inputs["reacher"].get("train_base_seed") != 20260841 \
            or inputs["reacher"].get("validation_base_seed") != 20260842 \
            or inputs["pusht"].get("split_seed") != 20260851 \
            or inputs["pusht"].get("start_seed") != 20260852 \
            or inputs["tworoom"].get("split_seed") != 20260861 \
            or inputs["tworoom"].get("start_seed") != 20260862:
        raise MatchedColorSpecError("fresh selection seeds changed")
    if inputs["reacher"].get("kind") != "fresh-dm-control-reacher-easy" \
            or inputs["reacher"].get("raw_frame_size") != 64 \
            or inputs["reacher"].get("target_alpha_zero") is not True \
            or inputs["pusht"].get("kind") != "official-root-hdf5" \
            or inputs["pusht"].get("state_key") != "state" \
            or inputs["tworoom"].get("kind") != "official-root-hdf5" \
            or inputs["tworoom"].get("state_key") != "observation":
        raise MatchedColorSpecError("input adapter contract changed")
    expected_v1_paths = {
        host: [
            f"outputs/paper_a_matched_host_v1/cache/{host}/base/train.npz",
            f"outputs/paper_a_matched_host_v1/cache/{host}/base/validation.npz"]
        for host in HDF_HOSTS}
    for host in HDF_HOSTS:
        if inputs[host].get("v1_base_cache_paths") != expected_v1_paths[host]:
            raise MatchedColorSpecError(f"V1 exclusion candidates changed: {host}")
        if inputs[host].get("v1_host_manifest_path") != (
                f"outputs/paper_a_matched_host_v1/cache/{host}/manifest.json"):
            raise MatchedColorSpecError(f"V1 host receipt path changed: {host}")

    outputs = _mapping(spec.get("outputs"), "outputs")
    root = resolve_path(outputs.get("root"))
    if outputs.get("root") != "outputs/paper_a_matched_color_v1":
        raise MatchedColorSpecError("Wave-1b output root changed")
    for key in ("cache", "carriers", "logs", "use"):
        child = resolve_path(outputs.get(key))
        if root not in child.parents:
            raise MatchedColorSpecError(f"outputs.{key} is outside output root")
    cache = _mapping(spec.get("cache"), "cache")
    if cache.get("frame_batch_size") != 128 \
            or cache.get("compression_level") != 1 \
            or cache.get("store_base_and_cue_separately") is not True:
        raise MatchedColorSpecError("cache contract changed")


def _verify_sidecar(spec_path: Path, sha_path: Path) -> str:
    if not sha_path.is_file():
        raise MatchedColorSpecError(f"missing protocol SHA sidecar: {sha_path}")
    fields = sha_path.read_text().strip().split()
    digest = sha256_file(spec_path)
    if len(fields) != 2 or fields[0] != digest or fields[1] != spec_path.name:
        raise MatchedColorSpecError("protocol SHA sidecar mismatch")
    return digest


def _verify_lock(spec: Mapping[str, Any], digest: str) -> dict[str, Any]:
    path = resolve_path(spec.get("implementation_lock"))
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise MatchedColorSpecError(f"cannot read implementation lock: {error}") from error
    if value.get("schema_version") != 1 \
            or value.get("study") != spec.get("study") \
            or value.get("spec_sha256") != digest \
            or value.get("adaptive_locked_before_formal_outcomes") is not True \
            or value.get("formal_output_absent_at_lock") is not True:
        raise MatchedColorSpecError("implementation lock identity mismatch")
    producers = value.get("producers")
    if not isinstance(producers, dict) or not producers:
        raise MatchedColorSpecError("implementation lock has no producers")
    for relative, expected in producers.items():
        source = resolve_path(relative)
        if not source.is_file() or sha256_file(source) != expected:
            raise MatchedColorSpecError(f"locked producer changed: {relative}")
    exclusions = _mapping(value.get("v1_hdf_exclusions"),
                          "implementation lock exclusions")
    for host in HDF_HOSTS:
        record = _mapping(exclusions.get(host), f"lock exclusions.{host}")
        indices = record.get("episode_indices")
        if not isinstance(indices, list) \
                or any(isinstance(item, bool) or not isinstance(item, int)
                       or item < 0 for item in indices) \
                or indices != sorted(set(indices)) \
                or record.get("count") != len(indices) \
                or record.get("indices_sha256") != _indices_sha256(indices):
            raise MatchedColorSpecError(f"invalid locked V1 exclusions: {host}")
        candidates = record.get("cache_candidates")
        if not isinstance(candidates, list) or len(candidates) != 2:
            raise MatchedColorSpecError(f"invalid exclusion receipts: {host}")
        expected_paths = spec["inputs"][host]["v1_base_cache_paths"]
        if [item.get("path") for item in candidates
                if isinstance(item, dict)] != expected_paths:
            raise MatchedColorSpecError(f"exclusion paths changed: {host}")
        for candidate in candidates:
            if candidate.get("present_at_lock") is not True:
                raise MatchedColorSpecError("all four V1 HDF caches must be pinned")
            source = resolve_path(candidate.get("path"))
            sidecar = resolve_path(candidate.get("sidecar_path"))
            if not source.is_file() \
                    or source.stat().st_size != candidate.get("size") \
                    or sha256_file(source) != candidate.get("sha256") \
                    or not sidecar.is_file() \
                    or sidecar.stat().st_size != candidate.get("sidecar_size") \
                    or sha256_file(sidecar) != candidate.get("sidecar_sha256"):
                raise MatchedColorSpecError(
                    f"locked V1 exclusion cache or sidecar changed: {source}")
        manifest = _mapping(record.get("v1_host_manifest"),
                            f"lock exclusions.{host}.v1_host_manifest")
        expected_manifest = spec["inputs"][host]["v1_host_manifest_path"]
        manifest_path = resolve_path(manifest.get("path"))
        if manifest.get("path") != expected_manifest \
                or not manifest_path.is_file() \
                or manifest_path.stat().st_size != manifest.get("size") \
                or sha256_file(manifest_path) != manifest.get("sha256"):
            raise MatchedColorSpecError(f"locked V1 host receipt changed: {host}")
    return {
        "path": str(path.relative_to(ROOT)), "sha256": sha256_file(path),
        "producers": len(producers), "v1_hdf_exclusions": exclusions,
    }


def load_locked_spec(spec_path: Path = DEFAULT_SPEC,
                     sha_path: Path = DEFAULT_SHA, *,
                     verify_inputs: bool = True) -> dict[str, Any]:
    spec_path, sha_path = spec_path.resolve(), sha_path.resolve()
    digest = _verify_sidecar(spec_path, sha_path)
    value = yaml.safe_load(spec_path.read_text())
    if not isinstance(value, dict):
        raise MatchedColorSpecError("protocol YAML must contain one mapping")
    validate_spec(value, verify_inputs=verify_inputs)
    result = dict(value)
    result["_lock"] = {
        "path": str(spec_path.relative_to(ROOT)), "sha256": digest,
        "sidecar": str(sha_path.relative_to(ROOT)),
        "sidecar_sha256": sha256_file(sha_path),
        "implementation": _verify_lock(result, digest),
    }
    return result


def v1_excluded_episode_indices(spec: Mapping[str, Any], host: str) -> tuple[int, ...]:
    if host not in HDF_HOSTS:
        return ()
    try:
        record = spec["_lock"]["implementation"]["v1_hdf_exclusions"][host]
    except (KeyError, TypeError) as error:
        raise MatchedColorSpecError("locked V1 exclusions are unavailable") from error
    return tuple(int(value) for value in record["episode_indices"])


def validate_device(spec: Mapping[str, Any], device: str) -> str:
    if device not in spec["execution"]["allowed_devices"]:
        raise MatchedColorSpecError(
            f"device {device!r} forbidden; only physical cuda:0 is admitted")
    return device


def output_path(spec: Mapping[str, Any], key: str) -> Path:
    if key not in ("root", "cache", "carriers", "logs", "use"):
        raise MatchedColorSpecError(f"unknown output key {key!r}")
    return resolve_path(spec["outputs"][key])


__all__ = [
    "AGES", "ALLOWED_DEVICES", "ARMS", "DEFAULT_LOCK", "DEFAULT_SHA",
    "DEFAULT_SPEC", "HDF_HOSTS", "HOSTS", "MatchedColorSpecError", "ROOT",
    "SEEDS", "TARGETS", "load_locked_spec", "output_path", "resolve_input_path",
    "resolve_path", "sha256_file", "v1_excluded_episode_indices",
    "validate_device", "validate_spec",
]
