#!/usr/bin/env python3
"""Post-hoc information-ceiling audit for a conserved V11 episode anchor.

This script does not train or alter a checkpoint.  It replays a completed KDIO-v11
checkpoint on immutable V11 caches and asks an evaluation-only question: how much task-state
information is linearly available when the strict pre-observation dynamic prior is augmented
with the visible initial encoder coordinate?

Four standardized clean-train ridge probes are fit:

``dynamic_prior``
    ``RMSNorm(q_t^- + v_t^-)`` only.  This must reproduce the checkpoint's saved
    ``heldout_prior_state_nmse``.
``initial_anchor``
    ``RMSNorm(z_0)`` only, repeated over target times.
``dynamic_plus_anchor``
    Concatenated ``[RMSNorm(q_t^- + v_t^-), RMSNorm(z_0)]``.
``dynamic_plus_shuffled_anchor``
    The same concatenation after a deterministic one-episode cyclic derangement of ``z_0``.

The task observation is consumed only by these post-training probes.  No simulator state,
task observation, corruption mask, or fitted probe feeds back into the model.  Evaluation uses
the exact V11 ``deep | first_post`` primary mask for the clean view and each held-out
corruption.  Source/cache/probe/feature hashes and sample counts make the diagnostic auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.hacssm_v11_data import (  # noqa: E402
    V11TrajectoryDataset,
    load_cache,
    sha256_file,
)
from scripts.train_hacssm_v10 import (  # noqa: E402
    _loader,
    _phase_masks as _v11_phase_masks,
    _probe_predict,
    _r2,
)
from scripts.train_hacssm_v11 import (  # noqa: E402
    HELDOUT_CONDITIONS,
    KDIO_DESIGNS,
    V11ExperimentModel,
    _fit_ridge,
    build_model,
)


SCHEMA_VERSION = 1
ANALYSIS_NAME = "hacssm_v11_conserved_initial_anchor_information_ceiling"
PROBE_NAMES = (
    "dynamic_prior",
    "initial_anchor",
    "dynamic_plus_anchor",
    "dynamic_plus_shuffled_anchor",
)
REPRODUCTION_ABS_TOLERANCE = 2e-4


class AuditError(RuntimeError):
    """Raised when an immutable-input or metric-reproduction contract fails."""


def _finite_float(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise AuditError(f"{label} must be finite, got {value!r}")
    return result


def _hash_arrays(values: Mapping[str, np.ndarray]) -> str:
    """Hash named arrays with explicit dtype and shape framing."""
    digest = hashlib.sha256()
    for name in sorted(values):
        array = np.ascontiguousarray(np.asarray(values[name]))
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _probe_hash(probe: Mapping[str, np.ndarray]) -> str:
    required = {"x_mean", "x_std", "y_mean", "y_std", "weights"}
    if set(probe) != required:
        raise AuditError(f"ridge probe fields {sorted(probe)} != {sorted(required)}")
    return _hash_arrays({name: np.asarray(probe[name]) for name in required})


def _source_hashes() -> dict[str, str]:
    paths = {
        "audit": Path(__file__).resolve(),
        "memory": ROOT / "lewm/models/memory.py",
        "memory_model": ROOT / "lewm/models/memory_model.py",
        "trainer": ROOT / "scripts/train_hacssm_v11.py",
        "data": ROOT / "scripts/hacssm_v11_data.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _metadata_record(metadata) -> dict[str, Any]:
    return {
        "path": str(metadata.path),
        "file_sha256": metadata.file_sha256,
        "content_sha256": metadata.content_sha256,
        "env_id": metadata.env_id,
        "split": metadata.split,
        "seed": metadata.seed,
        "episodes": metadata.episodes,
        "length": metadata.length,
        "img_size": metadata.img_size,
        "action_dim": metadata.action_dim,
        "task_observation_dim": metadata.task_observation_dim,
        "task_observation_keys": list(metadata.task_observation_keys),
        "task_observation_shapes": [list(shape) for shape in metadata.task_observation_shapes],
    }


def _validate_inputs(checkpoint: Mapping[str, Any], checkpoint_path: Path,
                     train_metadata, val_metadata) -> tuple[dict[str, Any], dict[str, Any]]:
    required = {
        "model_state_dict", "args", "final_metrics", "history", "state_probes",
        "inverse_action_probe", "action_history_probe",
    }
    if set(checkpoint) != required:
        raise AuditError(
            f"checkpoint fields {sorted(checkpoint)} != expected {sorted(required)}")
    saved_args = checkpoint["args"]
    metrics = checkpoint["final_metrics"]
    if not isinstance(saved_args, dict) or not isinstance(metrics, dict):
        raise AuditError("checkpoint args/final_metrics must be dictionaries")
    design = str(saved_args.get("memory_mode", ""))
    if design not in KDIO_DESIGNS:
        raise AuditError(f"anchor audit requires a KDIO-v11 checkpoint, got {design!r}")
    if metrics.get("design") != design:
        raise AuditError("checkpoint design disagrees with final metrics")
    if metrics.get("training_objective") is None:
        raise AuditError("checkpoint lacks a training-objective receipt")
    if train_metadata.split != "train" or val_metadata.split != "val":
        raise AuditError("expected immutable train/val V11 caches")
    train_schema = (
        train_metadata.env_id, train_metadata.length, train_metadata.img_size,
        train_metadata.action_dim, train_metadata.task_observation_dim,
        train_metadata.task_observation_keys, train_metadata.task_observation_shapes,
    )
    val_schema = (
        val_metadata.env_id, val_metadata.length, val_metadata.img_size,
        val_metadata.action_dim, val_metadata.task_observation_dim,
        val_metadata.task_observation_keys, val_metadata.task_observation_shapes,
    )
    if train_schema != val_schema:
        raise AuditError("train/validation cache schemas differ")
    expected_env = str(metrics.get("env", "")).removeprefix("dmc:")
    if train_metadata.env_id != expected_env:
        raise AuditError(
            f"cache environment {train_metadata.env_id!r} != checkpoint {expected_env!r}")
    for split, metadata in (("train", train_metadata), ("val", val_metadata)):
        saved_file_hash = metrics.get(f"{split}_data_sha256")
        saved_content_hash = metrics.get(f"{split}_data_content_sha256")
        if metadata.file_sha256 != saved_file_hash:
            raise AuditError(
                f"{split} cache SHA-256 differs from checkpoint receipt: "
                f"{metadata.file_sha256} != {saved_file_hash}")
        if metadata.content_sha256 != saved_content_hash:
            raise AuditError(f"{split} cache content SHA-256 differs from checkpoint receipt")
    if int(saved_args.get("history_len", -1)) < 1:
        raise AuditError("checkpoint has an invalid history length")
    if str(saved_args.get("eval_target_key")) != "task_observation":
        raise AuditError("anchor audit requires checkpoint eval_target_key='task_observation'")
    probes = checkpoint["state_probes"]
    if not isinstance(probes, dict) or "prior" not in probes:
        raise AuditError("checkpoint lacks its serialized prior probe")
    _probe_hash(probes["prior"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    return dict(saved_args), metrics


def _build_loaded_model(checkpoint: Mapping[str, Any], saved_args: dict[str, Any],
                        train_dataset: V11TrajectoryDataset,
                        device: torch.device) -> V11ExperimentModel:
    args = argparse.Namespace(**saved_args)
    args.device = str(device)
    action_mean = train_dataset.actions.mean(
        axis=(0, 1), dtype=np.float64).astype(np.float32)
    action_std = train_dataset.actions.std(
        axis=(0, 1), dtype=np.float64).clip(min=1e-6).astype(np.float32)
    model = build_model(
        args, train_dataset.metadata.action_dim, action_mean, action_std).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    if model.world.memory_impl not in KDIO_DESIGNS:
        raise AuditError(
            f"loaded model does not expose a KDIO recurrence: {model.world.memory_impl!r}")
    return model


@torch.no_grad()
def _collect(model: V11ExperimentModel, dataset: V11TrajectoryDataset,
             args: argparse.Namespace, device: torch.device,
             use_amp: bool) -> dict[str, np.ndarray]:
    """Collect episode-ordered strict priors, initial anchors, targets, and intervals."""
    model.eval()
    dynamic_chunks: list[np.ndarray] = []
    anchor_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    episode_chunks: list[np.ndarray] = []
    start_chunks: list[np.ndarray] = []
    end_chunks: list[np.ndarray] = []
    h = int(args.history_len)
    memory = model.world.mem_kdiov11
    for batch in _loader(dataset, args, train=False):
        if bool(batch["corruption_mask"][:, 0].any()):
            raise AuditError(
                f"{dataset.view}: initial frame is corrupted; z0 anchor is not legal")
        observed = batch["observed"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        amp_context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if use_amp else torch.autocast("cpu", enabled=False))
        with amp_context:
            z = model.world.encode(observed)
            _, details = memory(z, actions, return_details=True)
            # Deliberately bypass read_state: a future anchor-aware read may include c.  These
            # are the strict transition priors before z_t has corrected either q or v.
            dynamic = memory._rms_norm(
                details["q_priors"] + details["v_priors"])
            anchor = memory._rms_norm(z[:, 0])
        target = batch[args.eval_target_key]
        if dynamic.shape[:2] != target.shape[:2]:
            raise AuditError("dynamic-prior/target sequence shapes differ")
        dynamic_chunks.append(dynamic[:, h:].float().cpu().numpy())
        anchor_chunks.append(anchor.float().cpu().numpy())
        target_chunks.append(target[:, h:].float().numpy())
        episode_chunks.append(batch["episode_index"].numpy())
        start_chunks.append(batch["gap_start"].numpy())
        end_chunks.append(batch["gap_end"].numpy())

    result = {
        "dynamic": np.concatenate(dynamic_chunks),
        "anchor": np.concatenate(anchor_chunks),
        "target": np.concatenate(target_chunks),
        "episode_index": np.concatenate(episode_chunks),
        "gap_start": np.concatenate(start_chunks),
        "gap_end": np.concatenate(end_chunks),
    }
    order = np.argsort(result["episode_index"], kind="stable")
    result = {name: np.ascontiguousarray(value[order]) for name, value in result.items()}
    expected_episode_index = np.arange(len(dataset), dtype=result["episode_index"].dtype)
    if not np.array_equal(result["episode_index"], expected_episode_index):
        raise AuditError(f"{dataset.view}: episode ordering/coverage is not one-to-one")
    expected_steps = dataset.metadata.length - h
    if result["dynamic"].shape != (
            len(dataset), expected_steps, int(args.embed_dim)):
        raise AuditError(f"{dataset.view}: malformed dynamic-prior collection")
    if result["anchor"].shape != (len(dataset), int(args.embed_dim)):
        raise AuditError(f"{dataset.view}: malformed z0 collection")
    if result["target"].shape != (
            len(dataset), expected_steps, dataset.metadata.task_observation_dim):
        raise AuditError(f"{dataset.view}: malformed target collection")
    for name, value in result.items():
        if not np.isfinite(value).all():
            raise AuditError(f"{dataset.view}: collected {name} contains non-finite values")
    return result


def _repeated_anchor(bundle: Mapping[str, np.ndarray], *, shuffled: bool) -> np.ndarray:
    anchor = bundle["anchor"]
    if shuffled:
        if len(anchor) < 2:
            raise AuditError("cyclic anchor control requires at least two episodes")
        anchor = np.roll(anchor, shift=1, axis=0)
    steps = bundle["dynamic"].shape[1]
    return np.broadcast_to(anchor[:, None, :], (len(anchor), steps, anchor.shape[-1]))


def _features(bundle: Mapping[str, np.ndarray], probe_name: str) -> np.ndarray:
    dynamic = bundle["dynamic"]
    if probe_name == "dynamic_prior":
        value = dynamic
    elif probe_name == "initial_anchor":
        value = _repeated_anchor(bundle, shuffled=False)
    elif probe_name == "dynamic_plus_anchor":
        value = np.concatenate(
            (dynamic, _repeated_anchor(bundle, shuffled=False)), axis=-1)
    elif probe_name == "dynamic_plus_shuffled_anchor":
        value = np.concatenate(
            (dynamic, _repeated_anchor(bundle, shuffled=True)), axis=-1)
    else:
        raise AuditError(f"unknown probe {probe_name!r}")
    return np.ascontiguousarray(value.reshape(-1, value.shape[-1]), dtype=np.float32)


def _fit_probes(train_bundle: Mapping[str, np.ndarray], ridge: float
                ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, str]]:
    target = np.ascontiguousarray(
        train_bundle["target"].reshape(-1, train_bundle["target"].shape[-1]),
        dtype=np.float32)
    probes = {}
    feature_hashes = {}
    for name in PROBE_NAMES:
        feature = _features(train_bundle, name)
        probes[name] = _fit_ridge(feature, target, ridge)
        _probe_hash(probes[name])
        feature_hashes[name] = _hash_arrays({"features": feature, "target": target})
    return probes, feature_hashes


def _primary_masks(bundle: Mapping[str, np.ndarray], length: int,
                   history_len: int) -> dict[str, np.ndarray]:
    torch_masks = _v11_phase_masks(
        {
            "gap_start": torch.from_numpy(bundle["gap_start"]),
            "gap_end": torch.from_numpy(bundle["gap_end"]),
        },
        length,
        history_len,
        torch.device("cpu"),
    )
    deep = torch_masks["deep"]
    first_post = torch_masks["first_post"]
    return {
        "deep": deep.numpy(),
        "first_post": first_post.numpy(),
        "primary": (deep | first_post).numpy(),
    }


def _evaluate_probe(bundle: Mapping[str, np.ndarray], probe_name: str,
                    probe: Mapping[str, np.ndarray], length: int,
                    history_len: int, device: torch.device) -> dict[str, Any]:
    feature = _features(bundle, probe_name)
    target = np.ascontiguousarray(
        bundle["target"].reshape(-1, bundle["target"].shape[-1]), dtype=np.float32)
    prediction = _probe_predict(
        torch.from_numpy(feature).to(device), dict(probe)).float().cpu().numpy()
    y_std = np.asarray(probe["y_std"], dtype=np.float32)
    per_step = np.square((prediction - target) / y_std).mean(axis=-1)
    masks = _primary_masks(bundle, length, history_len)
    result: dict[str, Any] = {}
    flat_masks = {name: mask.reshape(-1) for name, mask in masks.items()}
    for phase, mask in flat_masks.items():
        if not bool(mask.any()):
            raise AuditError(f"evaluation mask {phase!r} is empty")
        result[f"{phase}_nmse"] = float(per_step[mask].mean(dtype=np.float64))
        result[f"{phase}_samples"] = int(mask.sum())
    primary = flat_masks["primary"]
    result["primary_r2"] = _r2(prediction[primary], target[primary])
    result["feature_sha256"] = _hash_arrays({"features": feature})
    for key, value in result.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise AuditError(f"non-finite evaluation result {key}")
    return result


def _evaluate_bundle(bundle: Mapping[str, np.ndarray],
                     probes: Mapping[str, Mapping[str, np.ndarray]],
                     length: int, history_len: int,
                     device: torch.device) -> dict[str, dict[str, Any]]:
    return {
        name: _evaluate_probe(
            bundle, name, probe, length, history_len, device)
        for name, probe in probes.items()
    }


def _dataset_counts(bundle: Mapping[str, np.ndarray], length: int,
                    history_len: int) -> dict[str, int]:
    masks = _primary_masks(bundle, length, history_len)
    result = {
        "episodes": int(bundle["dynamic"].shape[0]),
        "target_steps_per_episode": int(bundle["dynamic"].shape[1]),
        "probe_fit_or_eval_samples": int(
            bundle["dynamic"].shape[0] * bundle["dynamic"].shape[1]),
        "dynamic_feature_dim": int(bundle["dynamic"].shape[-1]),
        "anchor_feature_dim": int(bundle["anchor"].shape[-1]),
        "target_dim": int(bundle["target"].shape[-1]),
    }
    result.update({f"{name}_samples": int(mask.sum()) for name, mask in masks.items()})
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        help="completed KDIO-v11 model.pt")
    parser.add_argument("--train-data", required=True,
                        help="immutable V11 train cache used by the checkpoint")
    parser.add_argument("--val-data", required=True,
                        help="immutable V11 validation cache used by the checkpoint")
    parser.add_argument("--output", required=True,
                        help="new JSON receipt; existing files are never overwritten")
    parser.add_argument("--device", required=True,
                        help=(
                            "replay device, e.g. cuda:0; use the training device class for "
                            "exact verification"))
    return parser.parse_args()


def main() -> None:
    cli = _parse_args()
    checkpoint_path = Path(cli.checkpoint).resolve()
    output_path = Path(cli.output).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    device = torch.device(cli.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise AuditError(f"requested {device}, but CUDA is unavailable")
    if device.type not in {"cpu", "cuda"}:
        raise AuditError(f"unsupported analysis device {device}")

    train_metadata = load_cache(cli.train_data)
    val_metadata = load_cache(cli.val_data)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_args, saved_metrics = _validate_inputs(
        checkpoint, checkpoint_path, train_metadata, val_metadata)
    args = argparse.Namespace(**saved_args)
    use_amp = not bool(args.no_amp) and device.type == "cuda"

    train_dataset = V11TrajectoryDataset(
        train_metadata.path, "clean", int(args.corruption_seed), int(args.history_len))
    model = _build_loaded_model(checkpoint, saved_args, train_dataset, device)
    train_bundle = _collect(model, train_dataset, args, device, use_amp)
    probes, train_feature_hashes = _fit_probes(train_bundle, float(args.probe_ridge))

    results: dict[str, dict[str, dict[str, Any]]] = {}
    dataset_counts: dict[str, dict[str, int]] = {
        "train_clean": _dataset_counts(
            train_bundle, train_metadata.length, int(args.history_len))}
    dataset_hashes: dict[str, str] = {
        "train_clean": _hash_arrays(train_bundle)}
    del train_bundle, train_dataset
    saved_prior_probe = checkpoint["state_probes"]["prior"]
    replay_values: dict[str, float] = {}
    conditions = ("clean", *HELDOUT_CONDITIONS)
    for condition in conditions:
        dataset = V11TrajectoryDataset(
            val_metadata.path, condition, int(args.corruption_seed), int(args.history_len))
        bundle = _collect(model, dataset, args, device, use_amp)
        results[condition] = _evaluate_bundle(
            bundle, probes, val_metadata.length, int(args.history_len), device)
        dataset_counts[condition] = _dataset_counts(
            bundle, val_metadata.length, int(args.history_len))
        dataset_hashes[condition] = _hash_arrays(bundle)
        if condition in HELDOUT_CONDITIONS:
            replay_values[condition] = _evaluate_probe(
                bundle, "dynamic_prior", saved_prior_probe,
                val_metadata.length, int(args.history_len), device)["primary_nmse"]

    heldout_summary = {}
    for probe_name in PROBE_NAMES:
        values = [
            results[condition][probe_name]["primary_nmse"]
            for condition in HELDOUT_CONDITIONS
        ]
        heldout_summary[probe_name] = {
            "equal_condition_mean_primary_nmse": float(np.mean(values)),
            "condition_primary_nmse": {
                condition: results[condition][probe_name]["primary_nmse"]
                for condition in HELDOUT_CONDITIONS
            },
        }

    # Replay the checkpoint's serialized prior probe on the freshly extracted strict dynamic
    # coordinate. This separates feature-replay fidelity from fresh ridge-fit fidelity.
    replay_mean = float(np.mean(list(replay_values.values())))
    fresh_mean = heldout_summary["dynamic_prior"]["equal_condition_mean_primary_nmse"]
    saved_mean = _finite_float(
        saved_metrics.get("heldout_prior_state_nmse"),
        "saved heldout_prior_state_nmse")
    fresh_difference = abs(fresh_mean - saved_mean)
    replay_difference = abs(replay_mean - saved_mean)
    reproduction_passed = (
        fresh_difference <= REPRODUCTION_ABS_TOLERANCE
        and replay_difference <= REPRODUCTION_ABS_TOLERANCE)
    verification = {
        "saved_heldout_prior_state_nmse": saved_mean,
        "fresh_dynamic_prior_heldout_nmse": fresh_mean,
        "saved_probe_replay_heldout_nmse": replay_mean,
        "fresh_absolute_difference": fresh_difference,
        "saved_probe_replay_absolute_difference": replay_difference,
        "absolute_tolerance": REPRODUCTION_ABS_TOLERANCE,
        "passed": reproduction_passed,
        "saved_probe_replay_condition_nmse": replay_values,
    }
    if not reproduction_passed:
        raise AuditError(
            "strict dynamic-prior replay failed saved-metric reproduction: "
            f"fresh diff={fresh_difference:.8g}, saved-probe diff={replay_difference:.8g}, "
            f"tolerance={REPRODUCTION_ABS_TOLERANCE}")

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "analysis": ANALYSIS_NAME,
        "status": "posthoc_excluded_information_ceiling",
        "semantics": (
            "evaluation-only clean-train ridge; no model training, selection, or deployment claim"),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "design": saved_args["memory_mode"],
            "seed": int(saved_args["seed"]),
            "epochs": int(saved_args["epochs"]),
            "training_objective": saved_metrics["training_objective"],
            "saved_prior_probe_sha256": _probe_hash(saved_prior_probe),
        },
        "data": {
            "train": _metadata_record(train_metadata),
            "validation": _metadata_record(val_metadata),
        },
        "contract": {
            "dynamic_coordinate": "strict pre-observation RMSNorm(q_t^- + v_t^-)",
            "anchor_coordinate": "RMSNorm(encoder(observed_frame_t0))",
            "anchor_visibility_required": True,
            "probe_fit_split": "clean_train_all_t_ge_history",
            "probe_ridge": float(args.probe_ridge),
            "primary_mask": "deep | first_post",
            "heldout_conditions": list(HELDOUT_CONDITIONS),
            "cyclic_shuffle": "episode predecessor via numpy.roll(anchor, shift=1, axis=0)",
            "history_len": int(args.history_len),
            "embed_dim": int(args.embed_dim),
            "eval_target_key": args.eval_target_key,
            "use_bfloat16_amp": use_amp,
            "device": str(device),
        },
        "counts": dataset_counts,
        "hashes": {
            "source_sha256": _source_hashes(),
            "collected_array_sha256": dataset_hashes,
            "train_feature_target_sha256": train_feature_hashes,
            "fitted_probe_sha256": {
                name: _probe_hash(probe) for name, probe in probes.items()},
        },
        "results": results,
        "heldout_summary": heldout_summary,
        "verification": verification,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    serialized = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x") as stream:
        stream.write(serialized)
    print(json.dumps({
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "verification": verification,
        "heldout_summary": heldout_summary,
    }, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
