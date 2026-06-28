#!/usr/bin/env python3
"""Deterministic posthoc replay of HACSSM-v7 shrinkage endpoints.

This analyzer is deliberately outside the sealed primary V7 namespace.  It validates the
published study manifest and its referenced inputs, strictly reconstructs each of the 25 full-V7
models, and replays five fixed shrinkage conditions on all 150 validation episodes.  Clean targets
are passed only to the scoring stage, after every prediction has been materialized.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.data import PrecomputedFeatureDataset
from lewm.models.memory import HierarchicalActionConditionedMemory
from lewm.models.memory_model import MemoryLeWorldModel


OFFICIAL_MANIFEST_SHA256 = "98eda8abec229753381bed5f22c70317428242470cc6f40b6a3f9c16d0f55c11"
SEALED_ROOT = ROOT / "outputs" / "hacssm_v7_shared"
OUTPUT_PARENT = ROOT / "outputs"
CONDITION_ORDER = ("learned", "rho00", "rho11", "rho01", "rho10")
FIXED_RHOS = {
    "rho00": (0.0, 0.0),
    "rho11": (1.0, 1.0),
    "rho01": (0.0, 1.0),
    "rho10": (1.0, 0.0),
}
PHASE_ORDER = (
    "pre", "blackout_transition", "deep_blackout", "first_post", "recovery",
    "late_post", "all",
)


class EndpointReplayError(RuntimeError):
    """Fail-closed endpoint replay error."""


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise EndpointReplayError(f"required nonempty file is missing: {path}")
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def read_json(path: Path) -> Any:
    def reject_constant(token: str) -> None:
        raise ValueError(f"non-RFC JSON constant {token}")

    try:
        return json.loads(path.read_text(), parse_constant=reject_constant)
    except (OSError, UnicodeError, ValueError) as exc:
        raise EndpointReplayError(f"invalid JSON {path}: {exc}") from exc


def repo_path(relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise EndpointReplayError(f"unsafe repository-relative path: {relative!r}")
    resolved = (ROOT / value).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise EndpointReplayError(f"path escapes repository: {relative!r}") from exc
    return resolved


def verify_file_record(path: Path, expected: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(expected, Mapping):
        raise EndpointReplayError(f"{label}: malformed artifact record")
    if expected.get("kind", "file") != "file":
        raise EndpointReplayError(f"{label}: expected a regular file record")
    actual = file_record(path)
    if actual["bytes"] != expected.get("bytes") or actual["sha256"] != expected.get("sha256"):
        raise EndpointReplayError(
            f"{label}: artifact mismatch; actual={actual}, expected="
            f"{{'bytes': {expected.get('bytes')!r}, 'sha256': {expected.get('sha256')!r}}}")
    return actual


def input_path_key(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def snapshot_input(
    snapshot: dict[str, dict[str, Any]],
    path: Path,
    expected: Mapping[str, Any] | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    """Record one consumed input, optionally requiring a sealed expected record."""
    actual = (verify_file_record(path, expected, label or str(path))
              if expected is not None else file_record(path))
    key = input_path_key(path)
    prior = snapshot.get(key)
    if prior is not None and prior != actual:
        raise EndpointReplayError(f"input {key} was observed with conflicting records")
    snapshot[key] = actual
    return actual


def verify_input_snapshot(snapshot: Mapping[str, Mapping[str, Any]]) -> None:
    """Rehash every consumed input and require byte-for-byte pre/post identity."""
    if not isinstance(snapshot, Mapping) or not snapshot:
        raise EndpointReplayError("consumed-input snapshot is empty")
    for key, expected in sorted(snapshot.items()):
        path = Path(key) if Path(key).is_absolute() else ROOT / key
        verify_file_record(path, expected, f"post-replay input:{key}")


def validate_sealed_study(
    sealed_root: Path, expected_sha256: str,
) -> tuple[dict, dict, dict[str, dict[str, Any]]]:
    """Validate the immutable manifest, protocol, sealed sources, and feature artifacts."""
    sealed_root = sealed_root.resolve()
    manifest_path = sealed_root / "hacssm_v7_manifest.json"
    sidecar_path = sealed_root / "hacssm_v7_manifest.sha256"
    if len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256):
        raise EndpointReplayError("expected manifest SHA-256 must be 64 lowercase hex characters")
    actual_sha = sha256_file(manifest_path)
    if actual_sha != expected_sha256:
        raise EndpointReplayError(
            f"sealed manifest SHA-256 {actual_sha} != expected {expected_sha256}")
    wanted_sidecar = f"{actual_sha}  {manifest_path.name}\n"
    try:
        observed_sidecar = sidecar_path.read_text()
    except (OSError, UnicodeError) as exc:
        raise EndpointReplayError(f"cannot read manifest sidecar {sidecar_path}: {exc}") from exc
    if observed_sidecar != wanted_sidecar:
        raise EndpointReplayError(f"sealed manifest sidecar mismatch: {sidecar_path}")
    input_snapshot: dict[str, dict[str, Any]] = {}
    snapshot_input(input_snapshot, manifest_path)
    snapshot_input(input_snapshot, sidecar_path)

    manifest = read_json(manifest_path)
    required = {
        "schema_version", "completed_runs", "expected_runs", "all_requested_runs_completed",
        "producer_git_commit", "producer_git_clean", "protocol", "feature_artifacts",
        "eval_rollout_artifacts", "source_artifacts", "output_artifacts",
    }
    if not isinstance(manifest, dict) or not required.issubset(manifest):
        raise EndpointReplayError("sealed manifest is missing required fields")
    if (manifest["schema_version"] != 1 or manifest["completed_runs"] != 325
            or manifest["expected_runs"] != 325
            or manifest["all_requested_runs_completed"] is not True
            or manifest["producer_git_clean"] is not True):
        raise EndpointReplayError("sealed manifest does not describe the completed 325-cell study")

    protocol_entries = manifest["protocol"]
    if not isinstance(protocol_entries, dict) or len(protocol_entries) != 1:
        raise EndpointReplayError("sealed manifest must reference exactly one protocol")
    protocol_rel, protocol_record = next(iter(protocol_entries.items()))
    protocol_path = repo_path(protocol_rel)
    snapshot_input(input_snapshot, protocol_path, protocol_record, protocol_rel)
    protocol = read_json(protocol_path)
    if (not isinstance(protocol, dict)
            or protocol.get("producer_git_commit") != manifest["producer_git_commit"]
            or protocol.get("producer_git_clean") is not True
            or protocol.get("feature_artifacts") != manifest["feature_artifacts"]
            or protocol.get("eval_rollout_artifacts") != manifest["eval_rollout_artifacts"]
            or protocol.get("source_artifacts") != manifest["source_artifacts"]):
        raise EndpointReplayError("protocol and final manifest provenance differ")

    for collection_name in (
        "source_artifacts", "feature_artifacts", "eval_rollout_artifacts",
    ):
        records = manifest[collection_name]
        if not isinstance(records, dict) or not records:
            raise EndpointReplayError(f"sealed manifest has no {collection_name}")
        for relative, record in sorted(records.items()):
            snapshot_input(
                input_snapshot, repo_path(relative), record,
                f"{collection_name}:{relative}")
    return manifest, protocol, input_snapshot


def phase_indices(length: int, history_len: int) -> dict[str, np.ndarray]:
    if length <= history_len:
        raise EndpointReplayError("sequence length must exceed history length")
    target_times = np.arange(history_len, length, dtype=np.int64)
    occ_start = length // 3
    occ_end = min(length, occ_start + max(4, length // 5))
    deep_start = min(occ_end, occ_start + history_len)
    late_start = min(length, occ_end + history_len)
    masks = {
        "pre": target_times < occ_start,
        "blackout_transition": (target_times >= occ_start) & (target_times < deep_start),
        "deep_blackout": (target_times >= deep_start) & (target_times < occ_end),
        "first_post": target_times == occ_end,
        "recovery": (target_times > occ_end) & (target_times < late_start),
        "late_post": target_times >= late_start,
        "all": np.ones_like(target_times, dtype=np.bool_),
    }
    result = {}
    for name in PHASE_ORDER:
        indices = np.flatnonzero(masks[name])
        if indices.size == 0:
            raise EndpointReplayError(f"phase {name} has no targets")
        result[name] = indices
    return result


def memory_with_rho(
    memory,
    z: torch.Tensor,
    actions: torch.Tensor,
    rho: Sequence[float] | torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run the exact V7 recurrence with a fixed, externally supplied per-level shrinkage."""
    B, T, _ = memory._validate_latents(z)
    actions = memory._validate_actions(actions, B, T - 1).to(device=z.device, dtype=z.dtype)
    rho_tensor = torch.as_tensor(rho, device=z.device, dtype=z.dtype)
    if tuple(rho_tensor.shape) != (memory.K,):
        raise EndpointReplayError(f"rho shape {tuple(rho_tensor.shape)} != {(memory.K,)}")
    if not torch.isfinite(rho_tensor).all() or bool((rho_tensor < 0).any()) or bool((rho_tensor > 1).any()):
        raise EndpointReplayError("rho must be finite and lie in [0,1]")

    x = memory.W_x(z)
    initial = x[:, 0].unsqueeze(1).expand(-1, memory.K, -1)
    states = [initial]
    priors = [initial]
    static = torch.sigmoid(memory.gate_bias).view(1, memory.K, 1)
    rho_view = rho_tensor.view(1, memory.K, 1)

    dynamic_0 = HierarchicalActionConditionedMemory._dynamic_gate(
        memory, z[:, 0], x[:, 0], initial)
    gates = [(1.0 - rho_view) * static + rho_view * dynamic_0]
    state = initial
    beta = memory.betas.to(device=z.device, dtype=z.dtype).view(1, memory.K, 1)
    for time in range(1, T):
        prior = memory._action_prior_unchecked(state, actions[:, time - 1])
        dynamic = HierarchicalActionConditionedMemory._dynamic_gate(
            memory, z[:, time], x[:, time], prior)
        gate = (1.0 - rho_view) * static + rho_view * dynamic
        state = prior + beta * gate * (x[:, time].unsqueeze(1) - prior)
        priors.append(prior)
        states.append(state)
        gates.append(gate)

    state_sequence = torch.stack(states, dim=1)
    prior_sequence = torch.stack(priors, dim=1)
    gate_sequence = torch.stack(gates, dim=1)
    route = memory.route_weights().to(dtype=z.dtype)
    mixed = (state_sequence * route.view(1, 1, memory.K, 1)).sum(dim=2)
    mixed = mixed * torch.rsqrt(
        mixed.square().mean(dim=-1, keepdim=True) + memory.rms_eps)
    fused = memory.fuse(z, mixed)
    return fused, {
        "x": x,
        "priors": prior_sequence,
        "states": state_sequence,
        "gates": gate_sequence,
        "route": route,
    }


def predictor_windows(
    model: MemoryLeWorldModel, augmented: torch.Tensor, actions: torch.Tensor,
) -> torch.Tensor:
    B, length, dimension = augmented.shape
    history = model.history_len
    windows = length - history
    latent_windows = augmented.unfold(1, history, 1)[:, :windows]
    latent_windows = latent_windows.permute(0, 1, 3, 2).reshape(
        B * windows, history, dimension)
    action_windows = actions.unfold(1, history, 1)[:, :windows]
    action_windows = action_windows.permute(0, 1, 3, 2).reshape(
        B * windows, history, actions.shape[-1])
    prediction = model.predictor(latent_windows, action_windows)[:, -1]
    return prediction.reshape(B, windows, dimension)


def native_difference(
    native_fused: torch.Tensor,
    native_details: Mapping[str, torch.Tensor],
    replay_fused: torch.Tensor,
    replay_details: Mapping[str, torch.Tensor],
) -> float:
    differences = [(native_fused - replay_fused).abs().max()]
    if set(native_details) != set(replay_details):
        raise EndpointReplayError("native and replay detail schemas differ")
    for key in native_details:
        differences.append((native_details[key] - replay_details[key]).abs().max())
    return max(float(value) for value in differences)


@torch.inference_mode()
def predict_conditions(
    model: MemoryLeWorldModel,
    features_input: torch.Tensor,
    actions: torch.Tensor,
    batch_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Produce all condition predictions without accepting or reading any target tensor."""
    if model.memory_impl != "hacssmv7" or model.mem_hacssmv7.v7_mode != "dynamic":
        raise EndpointReplayError("endpoint replay accepts only the full hacssmv7 candidate")
    if features_input.dtype != torch.float32 or actions.dtype != torch.float32:
        raise EndpointReplayError("endpoint inference requires CPU float32 inputs")
    if features_input.device.type != "cpu" or actions.device.type != "cpu":
        raise EndpointReplayError("endpoint inference is CPU-only")
    if batch_size < 1:
        raise EndpointReplayError("batch size must be positive")

    memory = model.mem_hacssmv7
    learned_rho = memory.shrinkage().detach().cpu()
    condition_rhos = {"learned": learned_rho}
    condition_rhos.update({name: torch.tensor(value) for name, value in FIXED_RHOS.items()})
    chunks: dict[str, list[torch.Tensor]] = {name: [] for name in CONDITION_ORDER}
    maximum_native_difference = 0.0

    for start in range(0, features_input.shape[0], batch_size):
        stop = min(features_input.shape[0], start + batch_size)
        observed = features_input[start:stop]
        action_batch = actions[start:stop]
        z = model.encode(observed)
        native_fused, native_details = model._inject(
            z, actions=action_batch, return_memory_details=True)
        for condition in CONDITION_ORDER:
            replay_fused, replay_details = memory_with_rho(
                memory, z, action_batch, condition_rhos[condition])
            if condition == "learned":
                difference = native_difference(
                    native_fused, native_details, replay_fused, replay_details)
                maximum_native_difference = max(maximum_native_difference, difference)
                if difference != 0.0:
                    raise EndpointReplayError(
                        f"learned replay differs from native recurrence by {difference}")
            chunks[condition].append(
                predictor_windows(model, replay_fused, action_batch).cpu())

    predictions = {name: torch.cat(chunks[name], dim=0) for name in CONDITION_ORDER}
    weights = memory.W_a.weight.detach().reshape(
        memory.K, 2 * memory.embed_dim, memory.action_dim)
    norms = weights.flatten(1).norm(dim=1)
    cosine = F.cosine_similarity(weights[0].reshape(1, -1), weights[1].reshape(1, -1))[0]
    static = torch.sigmoid(memory.gate_bias.detach())
    route = memory.route_weights().detach()
    diagnostics = {
        "native_recurrence_max_abs": maximum_native_difference,
        "rho_fast": float(learned_rho[0]),
        "rho_medium": float(learned_rho[1]),
        "static_gate_fast": float(static[0]),
        "static_gate_medium": float(static[1]),
        "route_fast": float(route[0]),
        "route_medium": float(route[1]),
        "action_head_fast_norm": float(norms[0]),
        "action_head_medium_norm": float(norms[1]),
        "action_head_cosine": float(cosine),
    }
    return predictions, diagnostics


def score_predictions(
    predictions: Mapping[str, torch.Tensor],
    targets: torch.Tensor,
    history_len: int,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]]]:
    """Score already-materialized predictions against clean targets."""
    if targets.dtype != torch.float32 or targets.device.type != "cpu" or targets.dim() != 3:
        raise EndpointReplayError("targets must be a CPU float32 tensor (N,L,D)")
    phases = phase_indices(targets.shape[1], history_len)
    wanted_shape = (targets.shape[0], targets.shape[1] - history_len, targets.shape[2])
    clean = targets[:, history_len:]
    per_condition: dict[str, np.ndarray] = {}
    run_metrics: dict[str, dict[str, float]] = {}
    for condition in CONDITION_ORDER:
        prediction = predictions.get(condition)
        if prediction is None or tuple(prediction.shape) != wanted_shape:
            shape = None if prediction is None else tuple(prediction.shape)
            raise EndpointReplayError(
                f"condition {condition} prediction shape {shape} != {wanted_shape}")
        per_time = (prediction - clean).square().mean(dim=-1).numpy()
        if not np.isfinite(per_time).all() or np.any(per_time < 0):
            raise EndpointReplayError(f"condition {condition} produced invalid MSE")
        per_condition[condition] = per_time
        run_metrics[condition] = {
            f"mse_{phase}": float(per_time[:, indices].astype(np.float64).mean())
            for phase, indices in phases.items()
        }
    return per_condition, run_metrics


def load_validation_inputs(
    path: Path, feature_manifest_path: Path,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Load only observed features/actions and metadata; deliberately do not open clean targets."""
    manifest_sha = sha256_file(feature_manifest_path)
    required = {
        "schema_version", "split", "clean_env", "occ_env", "features_input",
        "features_target", "actions", "target_valid_mask", "n_actions", "constant_target",
        "feature_dim", "manifest_sha256",
    }
    with np.load(path, allow_pickle=False) as data:
        missing = required - set(data.files)
        if missing:
            raise EndpointReplayError(f"{path}: missing feature-cache fields {sorted(missing)}")
        if (int(data["schema_version"]) != 1 or str(data["split"]) != "val"
                or str(data["manifest_sha256"]) != manifest_sha):
            raise EndpointReplayError(f"{path}: validation-cache metadata mismatch")
        n_actions = int(data["n_actions"])
        feature_dim = int(data["feature_dim"])
        observed = np.array(data["features_input"], dtype=np.float32, copy=True)
        actions = np.array(data["actions"], dtype=np.int64, copy=True)
        target_mask = np.array(data["target_valid_mask"], dtype=np.bool_, copy=True)
    if observed.shape != (150, 32, 128) or feature_dim != 128:
        raise EndpointReplayError(f"{path}: unexpected observed-feature shape/protocol")
    if actions.shape != (150, 31) or n_actions != 6:
        raise EndpointReplayError(f"{path}: unexpected action shape/protocol")
    if actions.min() < 0 or actions.max() >= n_actions or not np.isfinite(observed).all():
        raise EndpointReplayError(f"{path}: invalid observed features/actions")
    expected_mask = np.ones(32, dtype=np.bool_)
    expected_mask[10:16] = False
    if not np.array_equal(target_mask, expected_mask):
        raise EndpointReplayError(f"{path}: unexpected target-valid phase mask")
    return observed, actions, n_actions


def build_model(checkpoint: Mapping[str, Any], n_actions: int) -> MemoryLeWorldModel:
    args = checkpoint["args"]
    if args.get("memory_mode") != "hacssmv7" or args.get("encoder_type") != "precomputed":
        raise EndpointReplayError("checkpoint is not a full precomputed-feature V7 model")
    if int(checkpoint["final_metrics"].get("n_actions", -1)) != n_actions:
        raise EndpointReplayError("checkpoint and validation cache action dimensions differ")
    torch.manual_seed(0)
    model = MemoryLeWorldModel(
        img_size=args["img_size"], patch_size=args["patch_size"],
        embed_dim=args["embed_dim"], action_dim=n_actions,
        encoder_layers=args["encoder_layers"], encoder_heads=args["encoder_heads"],
        predictor_layers=args["predictor_layers"], predictor_heads=args["predictor_heads"],
        predictor_norm=args["predictor_norm"], history_len=args["history_len"],
        dropout=args["dropout"], sigreg_lambda=args["sigreg_lambda"],
        sigreg_projections=args["sigreg_projections"], memory_mode="both",
        memory_impl="hacssmv7", tau_fast=args["tau_fast"], tau_slow=args["tau_slow"],
        learnable_alpha=not args["fixed_alpha"], smt_router=args["smt_router"],
        hier_loss_weight=args["hier_loss_weight"], encoder_type="precomputed",
    ).cpu().float()
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except (RuntimeError, TypeError, KeyError) as exc:
        raise EndpointReplayError(f"strict checkpoint load failed: {exc}") from exc
    model.eval()
    return model


def one_hot_actions(indices: np.ndarray, n_actions: int) -> torch.Tensor:
    action_indices = torch.from_numpy(np.array(indices, dtype=np.int64, copy=True))
    return F.one_hot(action_indices, num_classes=n_actions).to(dtype=torch.float32)


def condition_rho(condition: str, diagnostics: Mapping[str, float]) -> tuple[float, float]:
    if condition == "learned":
        return diagnostics["rho_fast"], diagnostics["rho_medium"]
    return FIXED_RHOS[condition]


def replay_job(
    job,
    manifest: Mapping[str, Any],
    batch_size: int,
    input_snapshot: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Validate, replay, then score one cell.  Target access occurs after prediction."""
    output_records = manifest["output_artifacts"]
    sealed_records = {}
    for path in (job.model_path, job.metrics_path, job.eval_rollout_path, job.wandb_run_path):
        relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
        if relative not in output_records:
            raise EndpointReplayError(f"sealed manifest has no artifact record for {relative}")
        sealed_records[relative] = snapshot_input(
            input_snapshot, path, output_records[relative], relative)

    checkpoint = torch.load(job.model_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
        "model_state_dict", "args", "final_metrics", "history",
    }:
        raise EndpointReplayError(f"unexpected checkpoint schema: {job.model_path}")
    args = checkpoint["args"]
    val_path = repo_path(args["val_feature_cache"])
    feature_manifest_path = repo_path(args["feature_manifest"])
    feature_records = manifest["feature_artifacts"]
    for path in (val_path, feature_manifest_path):
        relative = path.relative_to(ROOT.resolve()).as_posix()
        if relative not in feature_records:
            raise EndpointReplayError(f"sealed manifest has no feature record for {relative}")
        snapshot_input(input_snapshot, path, feature_records[relative], relative)
    observed_array, action_indices, n_actions = load_validation_inputs(
        val_path, feature_manifest_path)
    model = build_model(checkpoint, n_actions)
    observed = torch.from_numpy(observed_array)
    actions = one_hot_actions(action_indices, n_actions)
    predictions, diagnostics = predict_conditions(model, observed, actions, batch_size)

    # Full cache validation and clean-target access occur only after all predictions exist.
    dataset = PrecomputedFeatureDataset(str(val_path), str(feature_manifest_path))
    if (dataset.split != "val" or len(dataset) != 150 or dataset.features_input.shape != (150, 32, 128)
            or dataset.n_actions != n_actions
            or not np.array_equal(dataset.features_input, observed_array)
            or not np.array_equal(dataset.act, action_indices)):
        raise EndpointReplayError(f"post-prediction cache validation failed for {job.run_name}")
    targets = torch.from_numpy(np.array(dataset.features_target, dtype=np.float32, copy=True))
    per_time, metrics = score_predictions(predictions, targets, model.history_len)
    phases = phase_indices(targets.shape[1], model.history_len)
    checkpoint_relative = job.model_path.resolve().relative_to(ROOT.resolve()).as_posix()
    checkpoint_sha = sealed_records[checkpoint_relative]["sha256"]
    val_relative = val_path.relative_to(ROOT.resolve()).as_posix()
    val_sha = feature_records[val_relative]["sha256"]

    run_rows = []
    episode_rows = []
    for condition in CONDITION_ORDER:
        rho_fast, rho_medium = condition_rho(condition, diagnostics)
        run_row = {
            "run_name": job.run_name,
            "env": job.occ_env,
            "seed": int(job.seed),
            "condition": condition,
            **diagnostics,
            "rho_fast": rho_fast,
            "rho_medium": rho_medium,
            "episodes": len(dataset),
            "length": int(dataset.features_input.shape[1]),
            "history_len": int(model.history_len),
            "checkpoint_sha256": checkpoint_sha,
            "val_feature_sha256": val_sha,
            **metrics[condition],
        }
        run_rows.append(run_row)
        for episode in range(len(dataset)):
            row = {
                "run_name": job.run_name,
                "env": job.occ_env,
                "seed": int(job.seed),
                "episode": episode,
                "condition": condition,
                "rho_fast": rho_fast,
                "rho_medium": rho_medium,
            }
            for phase in PHASE_ORDER:
                row[f"mse_{phase}"] = float(
                    per_time[condition][episode, phases[phase]].astype(np.float64).mean())
            episode_rows.append(row)
    provenance = {
        "run_name": job.run_name,
        "checkpoint": {
            "path": checkpoint_relative,
            **sealed_records[checkpoint_relative],
        },
        "val_feature": {"path": val_relative, **file_record(val_path)},
        "feature_manifest": {
            "path": feature_manifest_path.relative_to(ROOT.resolve()).as_posix(),
            **file_record(feature_manifest_path),
        },
        "native_recurrence_max_abs": diagnostics["native_recurrence_max_abs"],
    }
    return run_rows, episode_rows, provenance


def summarize(run_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(run_rows) != 25 * len(CONDITION_ORDER):
        raise EndpointReplayError(f"expected 125 run-condition rows, got {len(run_rows)}")
    indexed = {
        (str(row["env"]), int(row["seed"]), str(row["condition"])): row
        for row in run_rows
    }
    if len(indexed) != len(run_rows):
        raise EndpointReplayError("duplicate run-condition rows")
    cells = sorted({(str(row["env"]), int(row["seed"])) for row in run_rows})
    if len(cells) != 25:
        raise EndpointReplayError(f"expected 25 cells, got {len(cells)}")

    conditions = {}
    contrasts = {}
    for condition in CONDITION_ORDER:
        condition_summary = {}
        for phase in PHASE_ORDER:
            key = f"mse_{phase}"
            environment_means = {
                env: float(np.mean([
                    float(indexed[e, seed, condition][key]) for e, seed in cells if e == env
                ], dtype=np.float64))
                for env in sorted({env for env, _ in cells})
            }
            condition_summary[phase] = {
                "environment_means": environment_means,
            }
        conditions[condition] = condition_summary
        if condition != "learned":
            phase_contrasts = {}
            for phase in PHASE_ORDER:
                key = f"mse_{phase}"
                learned = np.asarray(
                    [float(indexed[env, seed, "learned"][key]) for env, seed in cells],
                    dtype=np.float64)
                endpoint = np.asarray(
                    [float(indexed[env, seed, condition][key]) for env, seed in cells],
                    dtype=np.float64)
                if np.any(endpoint <= 0):
                    raise EndpointReplayError("endpoint MSE must be positive for relative contrast")
                relative = (endpoint - learned) / endpoint
                phase_contrasts[phase] = {
                    "mean_paired_relative_learned_advantage": float(relative.mean()),
                    "median_paired_relative_learned_advantage": float(np.median(relative)),
                    "learned_wins": int(np.sum(learned < endpoint)),
                    "ties": int(np.sum(learned == endpoint)),
                    "cells": len(cells),
                }
            contrasts[condition] = phase_contrasts

    envelope = {}
    for phase in PHASE_ORDER:
        key = f"mse_{phase}"
        learned = np.asarray(
            [float(indexed[env, seed, "learned"][key]) for env, seed in cells],
            dtype=np.float64)
        rho00 = np.asarray(
            [float(indexed[env, seed, "rho00"][key]) for env, seed in cells],
            dtype=np.float64)
        rho11 = np.asarray(
            [float(indexed[env, seed, "rho11"][key]) for env, seed in cells],
            dtype=np.float64)
        best = np.minimum(rho00, rho11)
        relative = (best - learned) / best
        envelope[phase] = {
            "mean_paired_relative_learned_advantage": float(relative.mean()),
            "median_paired_relative_learned_advantage": float(np.median(relative)),
            "learned_wins": int(np.sum(learned < best)),
            "ties": int(np.sum(learned == best)),
            "cells": len(cells),
        }

    learned_rows = [row for row in run_rows if row["condition"] == "learned"]
    diagnostics = {}
    for name in (
        "rho_fast", "rho_medium", "static_gate_fast", "static_gate_medium",
        "route_fast", "route_medium", "action_head_fast_norm",
        "action_head_medium_norm", "action_head_cosine",
    ):
        values = np.asarray([float(row[name]) for row in learned_rows], dtype=np.float64)
        diagnostics[name] = {
            "mean": float(values.mean()), "min": float(values.min()),
            "max": float(values.max()), "median": float(np.median(values)),
        }
    native_max = max(float(row["native_recurrence_max_abs"]) for row in learned_rows)
    return {
        "schema_version": 1,
        "analysis": "hacssm_v7_endpoint_replay",
        "cells": 25,
        "episodes_per_cell": 150,
        "conditions": list(CONDITION_ORDER),
        "primary_metric": "mse_first_post",
        "conditions_summary": conditions,
        "paired_contrasts_vs_learned": contrasts,
        "learned_vs_joint_endpoint_envelope": envelope,
        "learned_parameter_diagnostics": diagnostics,
        "native_recurrence_max_abs": native_max,
    }


RUN_FIELDS = (
    "run_name", "env", "seed", "condition", "rho_fast", "rho_medium", "episodes",
    "length", "history_len", "checkpoint_sha256", "val_feature_sha256",
    "native_recurrence_max_abs", "static_gate_fast", "static_gate_medium", "route_fast",
    "route_medium", "action_head_fast_norm", "action_head_medium_norm", "action_head_cosine",
    *(f"mse_{phase}" for phase in PHASE_ORDER),
)
EPISODE_FIELDS = (
    "run_name", "env", "seed", "episode", "condition", "rho_fast", "rho_medium",
    *(f"mse_{phase}" for phase in PHASE_ORDER),
)


def write_json_file(path: Path, value: Any) -> None:
    encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def write_csv_file(
    path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]],
) -> None:
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(fields), extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_results(
    output_parent: Path,
    sealed_manifest_sha256: str,
    sealed_root: Path,
    manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
    run_rows: Sequence[Mapping[str, Any]],
    episode_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    input_provenance: Sequence[Mapping[str, Any]],
    input_snapshot: Mapping[str, Mapping[str, Any]],
    batch_size: int,
) -> Path:
    """Publish a complete sibling directory with one atomic rename."""
    output_parent = output_parent.resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    name = f"hacssm_v7_endpoints_{sealed_manifest_sha256[:12]}"
    final = output_parent / name
    temporary = output_parent / f".{name}.{os.getpid()}.tmp"
    if final.exists():
        raise EndpointReplayError(f"refusing to overwrite endpoint output: {final}")
    if temporary.exists():
        raise EndpointReplayError(f"stale endpoint temporary directory: {temporary}")
    temporary.mkdir()
    try:
        per_run_path = temporary / "endpoint_per_run.csv"
        per_episode_path = temporary / "endpoint_per_episode.csv"
        summary_path = temporary / "summary.json"
        write_csv_file(per_run_path, RUN_FIELDS, run_rows)
        write_csv_file(per_episode_path, EPISODE_FIELDS, episode_rows)
        write_json_file(summary_path, summary)
        output_records = {
            path.name: file_record(path)
            for path in (per_run_path, per_episode_path, summary_path)
        }
        analyzer_path = Path(__file__).resolve()
        publication_snapshot = {
            str(key): dict(record) for key, record in input_snapshot.items()
        }
        analyzer_record = snapshot_input(publication_snapshot, analyzer_path)
        own_manifest = {
            "schema_version": 1,
            "analysis": "hacssm_v7_endpoint_replay",
            "sealed_study": {
                "root": sealed_root.resolve().relative_to(ROOT.resolve()).as_posix(),
                "manifest_sha256": sealed_manifest_sha256,
                "producer_git_commit": manifest["producer_git_commit"],
                "completed_runs": manifest["completed_runs"],
                "protocol_sha256": next(iter(manifest["protocol"].values()))["sha256"],
            },
            "analysis_source": {
                "path": analyzer_path.relative_to(ROOT.resolve()).as_posix(),
                **analyzer_record,
            },
            "sealed_source_artifacts": manifest["source_artifacts"],
            "sealed_feature_artifacts": manifest["feature_artifacts"],
            "sealed_eval_rollout_artifacts": manifest["eval_rollout_artifacts"],
            "configuration": {
                "conditions": list(CONDITION_ORDER),
                "fixed_rhos": {key: list(value) for key, value in FIXED_RHOS.items()},
                "device": "cpu",
                "dtype": "float32",
                "torch_deterministic_algorithms": True,
                "torch_num_threads": torch.get_num_threads(),
                "batch_size": batch_size,
                "cells": 25,
                "episodes_per_cell": 150,
                "target_access": "after_all_condition_predictions",
                "native_recurrence_required_exact": True,
            },
            "runtime": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "torch": torch.__version__,
            },
            "input_checkpoints": list(input_provenance),
            "consumed_input_artifacts": publication_snapshot,
            "native_recurrence_max_abs": summary["native_recurrence_max_abs"],
            "outputs": output_records,
            "protocol_analysis_contract": {
                "history_len": protocol["common_protocol"]["history_len"],
                "length": protocol["common_protocol"]["length"],
                "val_episodes": protocol["common_protocol"]["val_episodes"],
            },
        }
        own_manifest_path = temporary / "manifest.json"
        write_json_file(own_manifest_path, own_manifest)
        own_manifest_sha = sha256_file(own_manifest_path)
        sidecar = temporary / "manifest.sha256"
        with sidecar.open("xb") as stream:
            stream.write(f"{own_manifest_sha}  manifest.json\n".encode())
            stream.flush()
            os.fsync(stream.fileno())
        # This is intentionally the final read before publication.  Any checkpoint, feature,
        # protocol, sealed source, primary manifest, sidecar, or analyzer mutation aborts and the
        # temporary directory is removed without exposing a partial/stale result.
        verify_input_snapshot(publication_snapshot)
        fsync_directory(temporary)
        os.replace(temporary, final)
        fsync_directory(output_parent)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return final


def configure_determinism() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    np.random.seed(0)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed-root", type=Path, default=SEALED_ROOT)
    parser.add_argument("--output-parent", type=Path, default=OUTPUT_PARENT)
    parser.add_argument("--expected-manifest-sha256", default=OFFICIAL_MANIFEST_SHA256)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.batch_size < 1:
        raise EndpointReplayError("batch size must be positive")
    configure_determinism()
    manifest, protocol, input_snapshot = validate_sealed_study(
        args.sealed_root, args.expected_manifest_sha256)

    import scripts.run_hacssm_v7 as runner

    original_output_root = runner.OUTPUT_ROOT
    try:
        runner.OUTPUT_ROOT = args.sealed_root.resolve()
        runner.configure_shared()
        jobs = sorted(
            (job for job in runner.ALL_JOBS if job.design == "hacssmv7"),
            key=lambda job: (job.occ_env, job.seed),
        )
        if len(jobs) != 25:
            raise EndpointReplayError(f"expected 25 full-V7 jobs, got {len(jobs)}")
        all_run_rows = []
        all_episode_rows = []
        provenance = []
        for index, job in enumerate(jobs, 1):
            runner.shared.validate_job(job, allow_missing=False)
            run_rows, episode_rows, job_provenance = replay_job(
                job, manifest, args.batch_size, input_snapshot)
            all_run_rows.extend(run_rows)
            all_episode_rows.extend(episode_rows)
            provenance.append(job_provenance)
            print(f"[{index:02d}/25] replayed {job.run_name}", flush=True)
    finally:
        runner.OUTPUT_ROOT = original_output_root
        runner.configure_shared()

    summary = summarize(all_run_rows)
    final = publish_results(
        args.output_parent, args.expected_manifest_sha256, args.sealed_root,
        manifest, protocol, all_run_rows, all_episode_rows, summary, provenance,
        input_snapshot, args.batch_size,
    )
    result = {
        "output": final.relative_to(ROOT.resolve()).as_posix(),
        "manifest_sha256": sha256_file(final / "manifest.json"),
        "summary_sha256": sha256_file(final / "summary.json"),
        "native_recurrence_max_abs": summary["native_recurrence_max_abs"],
        "first_post": {
            condition: summary["conditions_summary"][condition]["first_post"]
            for condition in CONDITION_ORDER
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EndpointReplayError as exc:
        print(f"HACSSM-v7 endpoint replay error: {exc}", file=sys.stderr)
        raise SystemExit(2)
