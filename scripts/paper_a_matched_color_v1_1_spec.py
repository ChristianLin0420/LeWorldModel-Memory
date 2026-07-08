#!/usr/bin/env python3
"""Fail-closed loader for admission-informed matched-color Wave 1.1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "configs/paper_a_matched_color_v1_1.yaml"
DEFAULT_SHA = ROOT / "configs/paper_a_matched_color_v1_1.sha256"
DEFAULT_LOCK = ROOT / "configs/paper_a_matched_color_v1_1.lock.json"
HOSTS = ("reacher", "pusht")
TARGETS = ("color",)
AGES = (4, 8, 15)
ARMS = ("none", "gru", "lstm", "ssm", "fixed_trust")
SEEDS = (0, 1, 2, 3, 4)
ALLOWED_DEVICES = ("cuda:0",)
HDF_HOSTS = ("pusht",)


class MatchedColorV11SpecError(ValueError):
    """Wave 1.1 changed or is malformed."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def indices_sha256(values: list[int]) -> str:
    return hashlib.sha256(json.dumps(
        values, separators=(",", ":")).encode("ascii")).hexdigest()


def resolve_path(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise MatchedColorV11SpecError("path must be repository-relative")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise MatchedColorV11SpecError(f"path leaves repository: {value!r}")
    result = (ROOT / relative).resolve()
    try:
        result.relative_to(ROOT.resolve())
    except ValueError as error:
        raise MatchedColorV11SpecError(
            f"path leaves repository: {value!r}") from error
    return result


def resolve_input_path(record: Mapping[str, Any]) -> Path:
    if record.get("external_tmpfs") not in (None, False):
        raise MatchedColorV11SpecError("Wave 1.1 has no external input path")
    return resolve_path(record.get("path"))


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise MatchedColorV11SpecError(f"{label} must be a mapping")
    return value


def _exact(value: Any, expected: tuple[Any, ...], label: str) -> None:
    if value != list(expected):
        raise MatchedColorV11SpecError(
            f"{label} must be exactly {list(expected)!r}")


def _identity(record: Any, label: str, *, verify_hash: bool = True) -> Path:
    value = _mapping(record, label)
    path = resolve_path(value.get("path"))
    if not path.is_file() or path.stat().st_size != value.get("size"):
        raise MatchedColorV11SpecError(f"{label} file identity differs")
    digest = value.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise MatchedColorV11SpecError(f"{label} SHA is malformed")
    if verify_hash and sha256_file(path) != digest:
        raise MatchedColorV11SpecError(f"{label} SHA differs")
    return path


def _validate_prior_screens(spec: Mapping[str, Any]) -> None:
    origin = _mapping(spec.get("adaptive_origin"), "adaptive_origin")
    expected = {
        "deterministic_selection_rule": (
            "retain the unique Wave-1 target admitted at every registered age "
            "on both Reacher and PushT; keep its other factor as an exact-"
            "balanced nuisance"),
        "selected_target": "color",
        "retained_nuisance": "location",
        "selection_evidence": (
            "Wave-1 color passed every age on Reacher and PushT while location "
            "failed every age on both hosts"),
        "tworoom_boundary": (
            "TwoRoom is preserved as a stopped prerequisite in both prior "
            "screens and is not counted as a Wave 1.1 carrier environment"),
        "timing": (
            "declared after both prior admission screens and before every Wave "
            "1.1 admission or carrier outcome"),
        "prior_carrier_outcomes_observed": False,
        "prior_admission_metrics_used_for_selection": True,
        "prior_admission_metrics_used_for_wave1_1_inference": False,
        "preserve_both_prior_failures": True,
        "limitation": (
            "this adaptive matched-cue comparison reduces cue-semantic and "
            "evidence-age confounding but does not isolate checkpoint identity "
            "from environment, visual background, dynamics, or training "
            "distribution"),
        "hdf_exclusion_policy": (
            "union episode_index from identity-verified train and validation "
            "base caches of paper-a-matched-host-v1 and paper-a-matched-token-"
            "v1, pinned with sidecars and host receipts at Wave 1.1 seal time"),
        "reacher_rng_exclusion_policy": (
            "new train and validation RNG seeds must be disjoint from both prior "
            "screens and all three seed registries are hash-pinned at Wave 1.1 "
            "seal time"),
    }
    if any(origin.get(key) != value for key, value in expected.items()):
        raise MatchedColorV11SpecError("adaptive selection contract changed")
    screens = _mapping(origin.get("prior_screens"), "prior_screens")
    identities = {
        "matched_host_v1": (
            "paper-a-matched-host-v1", "stopped-admission-failure"),
        "matched_token_v1": ("paper-a-matched-token-v1", None),
    }
    for name, (study, _) in identities.items():
        screen = _mapping(screens.get(name), f"prior_screens.{name}")
        for key in ("spec", "sidecar", "implementation_lock"):
            _identity(screen.get(key), f"{name}.{key}")
        receipts = _mapping(screen.get("receipts"), f"{name}.receipts")
        for host in ("reacher", "pusht", "tworoom"):
            path = _identity(receipts.get(host), f"{name}.{host}.receipt")
            value = json.loads(path.read_text())
            if value.get("study") != study or value.get("host") != host:
                raise MatchedColorV11SpecError(
                    f"{name}/{host} receipt identity differs")
            expected_status = ("stopped-admission-failure" if name ==
                               "matched_host_v1" else
                               ("stopped-admission-failure" if host ==
                                "tworoom" else "admitted"))
            if value.get("status") != expected_status:
                raise MatchedColorV11SpecError(
                    f"{name}/{host} receipt status differs")
    v1 = json.loads(resolve_path(screens["matched_host_v1"][
        "receipts"]["reacher"]["path"]).read_text())
    pusht = json.loads(resolve_path(screens["matched_host_v1"][
        "receipts"]["pusht"]["path"]).read_text())
    for receipt in (v1, pusht):
        admission = _mapping(receipt.get("admission"), "Wave-1 admission")
        if not all(admission[f"age-{age}"]["color"]["admitted"] is True
                   and admission[f"age-{age}"]["location"]["admitted"] is False
                   for age in AGES):
            raise MatchedColorV11SpecError(
                "prior receipts do not imply the deterministic color rule")
    for screen in ("paper_a_matched_host_v1", "paper_a_matched_token_v1"):
        carrier_root = ROOT / "outputs" / screen / "carriers"
        if carrier_root.exists() and any(carrier_root.rglob("*")):
            raise MatchedColorV11SpecError(
                f"prior carrier outcome unexpectedly exists: {screen}")


def validate_spec(spec: Mapping[str, Any], *, verify_inputs: bool = False) -> None:
    if spec.get("schema_version") != 1 \
            or spec.get("study") != "paper-a-matched-color-v1-1" \
            or spec.get("protocol_status") \
            != "admission-informed-locked-before-formal-outcomes" \
            or spec.get("implementation_lock") \
            != "configs/paper_a_matched_color_v1_1.lock.json":
        raise MatchedColorV11SpecError("Wave 1.1 identity changed")
    _exact(spec.get("hosts"), HOSTS, "hosts")
    _exact(spec.get("targets"), TARGETS, "targets")
    _exact(spec.get("ages"), AGES, "ages")
    _exact(spec.get("arms"), ARMS, "arms")
    _exact(spec.get("seeds"), SEEDS, "seeds")
    _validate_prior_screens(spec)

    execution = _mapping(spec.get("execution"), "execution")
    _exact(execution.get("allowed_devices"), ALLOWED_DEVICES,
           "allowed_devices")
    if execution != {
            "allowed_devices": ["cuda:0"], "default_device": "cuda:0",
            "physical_gpu": 0, "explicit_execute_required": True,
            "deterministic_algorithms": True, "forbid_cuda3": True}:
        raise MatchedColorV11SpecError("GPU0 execution contract changed")

    sequence = _mapping(spec.get("sequence"), "sequence")
    if sequence != {
            "num_frames": 20, "frame_skip": 5, "decision_index": 19,
            "cue_length": 3,
            "cue_intervals": {"age-4": [12, 15], "age-8": [8, 11],
                              "age-15": [1, 4]},
            "predictor_history": 3, "endpoint_history": 3,
            "endpoint_feature": "concat(z[16],z[17],z[18],prior_read[19])",
            "current_observation_excluded": True}:
        raise MatchedColorV11SpecError("sequence contract changed")
    selection = _mapping(spec.get("selection"), "selection")
    if selection != {
            "train_episodes": 1200, "validation_episodes": 480,
            "one_sequence_per_episode": True,
            "episode_disjoint_splits": True,
            "exclude_all_prior_hdf_episode_indices_before_permutation": True,
            "train_label_seed": 20261031,
            "validation_label_seed": 20261032}:
        raise MatchedColorV11SpecError("fresh selection contract changed")
    cue = _mapping(spec.get("cue"), "cue")
    if cue != {
            "colors": 4, "locations": 4, "target": "color",
            "location_role": "exact-balanced randomized nuisance",
            "rgb_palette": [[230, 57, 70], [40, 160, 84], [45, 108, 223],
                            [239, 174, 45]],
            "normalized_centers": [[0.18, 0.18], [0.82, 0.18],
                                   [0.18, 0.82], [0.82, 0.82]],
            "square_side_fraction": 0.20,
            "all_16_combinations_exactly_balanced": True,
            "labels_independent_of_base_episode": True,
            "offcue_bytes_identical": True}:
        raise MatchedColorV11SpecError("simple color cue changed")
    admission = _mapping(spec.get("admission"), "admission")
    if admission != {
            "chance": 0.25, "cue_balanced_accuracy_min": 0.75,
            "cue_min_class_recall_min": 0.70, "shortcut_ceiling": 0.30,
            "shortcut_features": {
                "final_context_latent": "concat(z[16],z[17],z[18])",
                "final_action": "concat(a[15],a[16],a[17],a[18])",
                "final_state": "state[19]"},
            "all_hosts_ages_must_pass": True,
            "no_carrier_training_if_any_gate_fails": True,
            "frozen_host_hash_required": True}:
        raise MatchedColorV11SpecError("admission contract changed")
    training = _mapping(spec.get("carrier_training"), "carrier_training")
    if training != {
            "epochs": 100, "batch_size": 64, "learning_rate": 0.0003,
            "weight_decay": 0.00001, "scheduler": "cosine-annealing",
            "gradient_clip_norm": 1.0, "windows_per_episode": 8,
            "training_rng_offset": 571000, "age_balanced_mixture": True,
            "frozen_encoder": True, "frozen_predictor": True}:
        raise MatchedColorV11SpecError("carrier training changed")
    readout = _mapping(spec.get("readout"), "readout")
    if readout != {
            "model": "StandardScaler+LogisticRegression", "logistic_c": 1.0,
            "solver": "lbfgs", "max_iter": 4000, "random_state": 0,
            "metric": "balanced_accuracy", "separate_readout_per_age": True}:
        raise MatchedColorV11SpecError("readout changed")
    statistics = _mapping(spec.get("statistics"), "statistics")
    if statistics != {
            "bootstrap_draws": 20000, "bootstrap_seed": 20261021,
            "confidence": 0.95, "interval": "percentile",
            "seed_resampling": "jointly resample carrier seed indices across hosts",
            "episode_resampling": (
                "independently by host and stratified by 16-way joint label"),
            "preserve_pairing": "ages and arms within host",
            "equivalence_margin": 0.05,
            "primary_interaction": (
                "age-15 PushT-minus-Reacher difference in fixed-trust-minus-"
                "SSM color balanced accuracy"),
            "nuisance_location_summary": (
                "per-location accuracy and worst-location accuracy for every "
                "host-arm-age cell"),
            "no_pooled_host_memory_score": True}:
        raise MatchedColorV11SpecError("statistical contract changed")
    action = _mapping(spec.get("action_normalization"), "action_normalization")
    if action != {
            "source": "all finite raw two-dimensional training-source actions per host",
            "ddof": 1,
            "order": (
                "normalize each raw 2-D action then time-major flatten five "
                "actions to 10-D")}:
        raise MatchedColorV11SpecError("action normalization changed")
    if spec.get("cache") != {
            "frame_batch_size": 128, "compression_level": 1,
            "store_base_and_cue_separately": True}:
        raise MatchedColorV11SpecError("cache contract changed")

    inputs = _mapping(spec.get("inputs"), "inputs")
    reacher = _mapping(inputs.get("reacher"), "inputs.reacher")
    pusht = _mapping(inputs.get("pusht"), "inputs.pusht")
    if reacher.get("train_base_seed") != 20261041 \
            or reacher.get("validation_base_seed") != 20261042 \
            or reacher.get("prior_seed_pairs") \
            != [[20260741, 20260742], [20260941, 20260942]] \
            or set((20261041, 20261042)).intersection(
                value for pair in reacher["prior_seed_pairs"] for value in pair):
        raise MatchedColorV11SpecError("Reacher fresh RNG contract changed")
    if pusht.get("split_seed") != 20261051 \
            or pusht.get("start_seed") != 20261052 \
            or pusht.get("prior_base_cache_paths") != [
                "outputs/paper_a_matched_host_v1/cache/pusht/base/train.npz",
                "outputs/paper_a_matched_host_v1/cache/pusht/base/validation.npz",
                "outputs/paper_a_matched_token_v1/cache/pusht/base/train.npz",
                "outputs/paper_a_matched_token_v1/cache/pusht/base/validation.npz"]:
        raise MatchedColorV11SpecError("PushT fresh selection changed")
    expected_sources = {
        "reacher": ("quentinll/lewm-reacher",
                    "62adae4b71dc474ddf8f794c476ebfe737a743ca", None,
                    ("weights",)),
        "pusht": ("quentinll/lewm-pusht",
                  "22b330c28c27ead4bfd1888615af1340e3fe9052",
                  "655cd446b9929369d7d406001da85c15d1457850",
                  ("config", "weights", "dataset")),
    }
    for host in HOSTS:
        value = _mapping(inputs.get(host), f"inputs.{host}")
        repo, revision, dataset_revision, identity_names = expected_sources[host]
        if value.get("repo_id") != repo or value.get("model_revision") != revision \
                or (dataset_revision is not None and
                    value.get("dataset_revision") != dataset_revision):
            raise MatchedColorV11SpecError(f"{host} source identity changed")
        _exact(value.get("identity_records"), identity_names,
               f"{host}.identity_records")
        for key in identity_names:
            _identity(value.get(key), f"inputs.{host}.{key}",
                      verify_hash=verify_inputs)
    if reacher.get("kind") != "fresh-dm-control-reacher-easy" \
            or reacher.get("raw_frame_size") != 64 \
            or reacher.get("target_alpha_zero") is not True \
            or pusht.get("kind") != "official-root-hdf5" \
            or pusht.get("state_key") != "state":
        raise MatchedColorV11SpecError("host adapter changed")
    outputs = _mapping(spec.get("outputs"), "outputs")
    if outputs.get("root") != "outputs/paper_a_matched_color_v1_1":
        raise MatchedColorV11SpecError("output root changed")
    root = resolve_path(outputs["root"])
    for key in ("cache", "carriers", "logs"):
        child = resolve_path(outputs.get(key))
        if root not in child.parents:
            raise MatchedColorV11SpecError(f"outputs.{key} leaves root")


def _verify_sidecar(spec_path: Path, sha_path: Path) -> str:
    fields = sha_path.read_text().strip().split()
    digest = sha256_file(spec_path)
    if len(fields) != 2 or fields != [digest, spec_path.name]:
        raise MatchedColorV11SpecError("protocol SHA sidecar differs")
    return digest


def _verify_lock(spec: Mapping[str, Any], digest: str) -> dict[str, Any]:
    path = resolve_path(spec.get("implementation_lock"))
    value = json.loads(path.read_text())
    if value.get("schema_version") != 1 \
            or value.get("study") != spec.get("study") \
            or value.get("spec_sha256") != digest \
            or value.get("locked_before_wave1_1_outcomes") is not True \
            or value.get("formal_output_absent_at_lock") is not True:
        raise MatchedColorV11SpecError("implementation lock identity differs")
    producers = _mapping(value.get("producers"), "lock.producers")
    for relative, expected in producers.items():
        source = resolve_path(relative)
        if not source.is_file() or sha256_file(source) != expected:
            raise MatchedColorV11SpecError(f"locked producer changed: {relative}")
    exclusions = _mapping(value.get("prior_hdf_exclusions"),
                          "lock.prior_hdf_exclusions")
    record = _mapping(exclusions.get("pusht"), "PushT exclusions")
    indices = record.get("episode_indices")
    if not isinstance(indices, list) or indices != sorted(set(indices)) \
            or record.get("count") != len(indices) \
            or record.get("indices_sha256") != indices_sha256(indices) \
            or record.get("count") != 3360:
        raise MatchedColorV11SpecError("PushT exclusion union differs")
    rng = _mapping(value.get("reacher_rng_exclusion"),
                   "lock.reacher_rng_exclusion")
    if rng.get("new_seeds") != [20261041, 20261042] \
            or rng.get("prior_seeds") \
            != [20260741, 20260742, 20260941, 20260942] \
            or set(rng["new_seeds"]).intersection(rng["prior_seeds"]):
        raise MatchedColorV11SpecError("Reacher RNG exclusion differs")
    return {
        "path": str(path.relative_to(ROOT)), "sha256": sha256_file(path),
        "producers": len(producers),
        "prior_hdf_exclusions": exclusions,
        "reacher_rng_exclusion": rng,
        # Compatibility alias used by the locked, reused color preparer.
        "v1_hdf_exclusions": exclusions,
    }


def load_locked_spec(spec_path: Path = DEFAULT_SPEC,
                     sha_path: Path = DEFAULT_SHA, *,
                     verify_inputs: bool = True) -> dict[str, Any]:
    digest = _verify_sidecar(spec_path.resolve(), sha_path.resolve())
    value = yaml.safe_load(spec_path.read_text())
    if not isinstance(value, dict):
        raise MatchedColorV11SpecError("protocol YAML is not a mapping")
    validate_spec(value, verify_inputs=verify_inputs)
    result = dict(value)
    result["_lock"] = {
        "path": str(spec_path.resolve().relative_to(ROOT)),
        "sha256": digest,
        "sidecar": str(sha_path.resolve().relative_to(ROOT)),
        "sidecar_sha256": sha256_file(sha_path),
        "implementation": _verify_lock(result, digest),
    }
    return result


def prior_excluded_episode_indices(spec: Mapping[str, Any], host: str) -> tuple[int, ...]:
    if host not in HDF_HOSTS:
        raise MatchedColorV11SpecError(f"no HDF exclusions for {host}")
    return tuple(spec["_lock"]["implementation"][
        "prior_hdf_exclusions"][host]["episode_indices"])


def validate_device(spec: Mapping[str, Any], device: str) -> str:
    if device not in spec["execution"]["allowed_devices"]:
        raise MatchedColorV11SpecError("Wave 1.1 requires logical cuda:0")
    return device


def output_path(spec: Mapping[str, Any], key: str) -> Path:
    if key not in ("root", "cache", "carriers", "logs"):
        raise MatchedColorV11SpecError(f"unknown output key {key}")
    return resolve_path(spec["outputs"][key])


__all__ = [
    "AGES", "ALLOWED_DEVICES", "ARMS", "DEFAULT_LOCK", "DEFAULT_SHA",
    "DEFAULT_SPEC", "HDF_HOSTS", "HOSTS", "SEEDS", "TARGETS",
    "indices_sha256", "load_locked_spec", "output_path",
    "prior_excluded_episode_indices", "resolve_input_path", "resolve_path",
    "sha256_file", "validate_device", "validate_spec",
]
