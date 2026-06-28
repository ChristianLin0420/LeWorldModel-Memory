#!/usr/bin/env python3
"""Precompute paired DINOv2/PCA features from schema-v3 clean robot caches.

Only clean pixels are passed through DINOv2.  The occluded input feature stream is
derived by replacing the fixed blackout interval with the feature produced by the
same exact preprocessing applied to an all-black frame.  PCA is fit without whitening
on *visible clean training frames only*; validation pixels never influence the PCA.

The two consumer NPZ files intentionally have a small exact schema.  All other
provenance, quality measurements, PCA hashes, source-cache hashes, software versions,
and model fingerprints live in the JSON manifest.  Each NPZ embeds the SHA-256 of the
exact manifest file bytes expected by the training consumer.  Existing artifacts are
resumed only after validating that hash, source/model provenance, NPZ schemas, and
every stored array-content hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"
RAW_FEATURE_DIM = 384
PIXEL_SCHEMA_VERSION = 3
FEATURE_SCHEMA_VERSION = 1
PRODUCER_PROTOCOL_VERSION = 1
PROTOTYPE_SEED = 0
IMG_SIZE = 64
TRAIN_ROLLOUT_SEED = 0
VAL_ROLLOUT_SEED = 7777
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

NPZ_FIELDS = frozenset(
    {
        "schema_version",
        "split",
        "clean_env",
        "occ_env",
        "features_input",
        "features_target",
        "actions",
        "target_valid_mask",
        "n_actions",
        "constant_target",
        "feature_dim",
        "manifest_sha256",
    }
)
NPZ_CONTENT_FIELDS = NPZ_FIELDS - {"manifest_sha256"}


@dataclass
class SplitPixels:
    split: str
    clean_obs: np.ndarray
    actions: np.ndarray
    action_prototypes: np.ndarray
    n_actions: int
    sources: Dict[str, Dict[str, Any]]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def scalar_value(value: np.ndarray, label: str) -> Any:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{label} must be a scalar, got shape {array.shape}")
    return array.item()


def finite_float(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is non-finite: {value!r}")
    return result


def safe_env_name(env: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in env).strip("_")


def cache_path(data_dir: Path, env: str, episodes: int, length: int, seed: int) -> Path:
    filename = (
        f"{env.replace(':', '_')}_v{PIXEL_SCHEMA_VERSION}_proto{PROTOTYPE_SEED}_"
        f"n{episodes}_L{length}_s{IMG_SIZE}_seed{seed}.npz"
    )
    return data_dir / filename


def cache_record(path: Path, data: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    stat = path.stat()
    record: Dict[str, Any] = {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": file_sha256(path),
        "schema_version": int(scalar_value(data["schema_version"], f"{path}:schema_version")),
        "prototype_seed": int(scalar_value(data["prototype_seed"], f"{path}:prototype_seed")),
        "cache_role": str(scalar_value(data["cache_role"], f"{path}:cache_role")),
        "n_actions": int(scalar_value(data["n_actions"], f"{path}:n_actions")),
        "action_prototypes_sha256": array_sha256(np.asarray(data["action_prototypes"])),
        "actions_sha256": array_sha256(np.asarray(data["actions"])),
        "observations_sha256": array_sha256(np.asarray(data["obs"])),
    }
    if "clean_env_id" in data:
        record["clean_env_id"] = str(
            scalar_value(data["clean_env_id"], f"{path}:clean_env_id")
        )
    return record


def load_split_pixels(
    clean_env: str,
    data_dir: Path,
    episodes: int,
    length: int,
    seed: int,
    split: str,
) -> SplitPixels:
    occ_env = f"{clean_env}.occ"
    clean_path = cache_path(data_dir, clean_env, episodes, length, seed)
    occ_path = cache_path(data_dir, occ_env, episodes, length, seed)
    for path in (clean_path, occ_path):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"missing/nonempty schema-v3 pixel cache: {path}")

    required = {
        "obs",
        "actions",
        "n_actions",
        "action_prototypes",
        "prototype_seed",
        "schema_version",
        "cache_role",
    }
    with np.load(clean_path, allow_pickle=False) as clean, np.load(
        occ_path, allow_pickle=False
    ) as occ:
        clean_missing = required - set(clean.files)
        occ_missing = (required | {"clean_env_id"}) - set(occ.files)
        if clean_missing or occ_missing:
            raise ValueError(
                f"cache metadata missing: clean={sorted(clean_missing)}, occ={sorted(occ_missing)}"
            )
        clean_schema = int(scalar_value(clean["schema_version"], "clean schema"))
        occ_schema = int(scalar_value(occ["schema_version"], "occ schema"))
        if clean_schema != PIXEL_SCHEMA_VERSION or occ_schema != PIXEL_SCHEMA_VERSION:
            raise ValueError(
                f"pixel schema mismatch: clean={clean_schema}, occ={occ_schema}, "
                f"expected {PIXEL_SCHEMA_VERSION}"
            )
        clean_prototype_seed = int(scalar_value(clean["prototype_seed"], "clean prototype"))
        occ_prototype_seed = int(scalar_value(occ["prototype_seed"], "occ prototype"))
        if (clean_prototype_seed, occ_prototype_seed) != (PROTOTYPE_SEED, PROTOTYPE_SEED):
            raise ValueError("pixel caches do not use prototype_seed=0")
        if str(scalar_value(clean["cache_role"], "clean role")) != "clean_or_full":
            raise ValueError(f"{clean_path}: expected cache_role='clean_or_full'")
        if str(scalar_value(occ["cache_role"], "occ role")) != "paired_occluded":
            raise ValueError(f"{occ_path}: expected cache_role='paired_occluded'")
        if str(scalar_value(occ["clean_env_id"], "occ clean_env_id")) != clean_env:
            raise ValueError(f"{occ_path}: clean_env_id does not match {clean_env!r}")

        clean_obs = np.asarray(clean["obs"])
        occ_obs = np.asarray(occ["obs"])
        clean_actions = np.asarray(clean["actions"])
        occ_actions = np.asarray(occ["actions"])
        clean_prototypes = np.asarray(clean["action_prototypes"])
        occ_prototypes = np.asarray(occ["action_prototypes"])
        clean_n_actions = int(scalar_value(clean["n_actions"], "clean n_actions"))
        occ_n_actions = int(scalar_value(occ["n_actions"], "occ n_actions"))

        expected_obs_shape = (episodes, length, IMG_SIZE, IMG_SIZE, 3)
        expected_action_shape = (episodes, length - 1)
        if clean_obs.shape != expected_obs_shape or occ_obs.shape != expected_obs_shape:
            raise ValueError(
                f"observation shape mismatch: clean={clean_obs.shape}, occ={occ_obs.shape}, "
                f"expected {expected_obs_shape}"
            )
        if clean_obs.dtype != np.uint8 or occ_obs.dtype != np.uint8:
            raise ValueError("pixel caches must store uint8 observations")
        if clean_actions.shape != expected_action_shape or occ_actions.shape != expected_action_shape:
            raise ValueError("pixel-cache action shapes do not match the requested split")
        if not np.issubdtype(clean_actions.dtype, np.integer):
            raise ValueError("pixel-cache actions must be integer-valued")
        if clean_n_actions <= 0 or clean_n_actions != occ_n_actions:
            raise ValueError("pixel-cache n_actions mismatch")
        if not np.array_equal(clean_actions, occ_actions):
            raise ValueError("clean/occluded action trajectories are not identical")
        if not np.array_equal(clean_prototypes, occ_prototypes):
            raise ValueError("clean/occluded action prototypes are not identical")
        if not np.isfinite(clean_prototypes).all():
            raise ValueError("action prototypes contain non-finite values")
        if clean_actions.min() < 0 or clean_actions.max() >= clean_n_actions:
            raise ValueError("action index outside n_actions")

        occ_start = length // 3
        occ_end = min(length, occ_start + max(4, length // 5))
        exact_masked_copy = (
            np.array_equal(occ_obs[:, :occ_start], clean_obs[:, :occ_start])
            and not np.any(occ_obs[:, occ_start:occ_end])
            and np.array_equal(occ_obs[:, occ_end:], clean_obs[:, occ_end:])
        )
        if not exact_masked_copy:
            raise ValueError(
                f"{split}: .occ cache is not an exact masked copy in [{occ_start},{occ_end})"
            )
        if not np.any(clean_obs[:, occ_start:occ_end]):
            raise ValueError(f"{split}: clean blackout interval is entirely black")

        sources = {
            f"{split}_clean": cache_record(clean_path, clean),
            f"{split}_occ": cache_record(occ_path, occ),
        }
        # Arrays loaded from an NPZ are standalone ndarrays, not mmap views, so they
        # remain valid after the handles close. Avoid duplicating the large clean pixels.
        return SplitPixels(
            split=split,
            clean_obs=np.ascontiguousarray(clean_obs),
            actions=np.asarray(clean_actions, dtype=np.int64).copy(),
            action_prototypes=np.array(clean_prototypes, copy=True),
            n_actions=clean_n_actions,
            sources=sources,
        )


def state_fingerprint(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    state = model.state_dict()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def load_dino(device: torch.device) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    import timm

    model = timm.create_model(
        MODEL_NAME,
        pretrained=True,
        num_classes=0,
        img_size=224,
        dynamic_img_size=True,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if int(getattr(model, "num_features", -1)) != RAW_FEATURE_DIM:
        raise ValueError(
            f"{MODEL_NAME} num_features={getattr(model, 'num_features', None)}, "
            f"expected {RAW_FEATURE_DIM}"
        )
    fingerprint = state_fingerprint(model)
    metadata = {
        "model_name": MODEL_NAME,
        "raw_feature_dim": RAW_FEATURE_DIM,
        "state_fingerprint_sha256": fingerprint,
        "preprocessing": {
            "input": "uint8 RGB divided by 255",
            "resize": "torch bilinear 224x224 align_corners=False",
            "normalization_mean": list(IMAGENET_MEAN),
            "normalization_std": list(IMAGENET_STD),
        },
    }
    return model.to(device), metadata


@torch.inference_mode()
def encode_clean_frames(
    model: torch.nn.Module,
    observations: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    flat = observations.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    result = np.empty((flat.shape[0], RAW_FEATURE_DIM), dtype=np.float32)
    mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    for start in range(0, len(flat), batch_size):
        pixels = torch.from_numpy(np.ascontiguousarray(flat[start : start + batch_size]))
        pixels = pixels.to(device=device, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        pixels = F.interpolate(
            pixels,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )
        pixels = (pixels - mean) / std
        features = model(pixels)
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise TypeError(f"unexpected DINO output type/shape: {type(features)}, {features!r}")
        if features.shape != (pixels.shape[0], RAW_FEATURE_DIM):
            raise ValueError(f"unexpected DINO output shape {tuple(features.shape)}")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("DINO produced non-finite features")
        result[start : start + len(features)] = features.float().cpu().numpy()
    return result.reshape(*observations.shape[:2], RAW_FEATURE_DIM)


def encode_black_frame(
    model: torch.nn.Module,
    device: torch.device,
) -> np.ndarray:
    black = np.zeros((1, 1, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    return encode_clean_frames(model, black, device, batch_size=1)[0, 0]


def covariance_quality(features: np.ndarray, label: str) -> Dict[str, Any]:
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError(f"{label}: expected an N x D matrix with N>=2, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{label}: non-finite feature matrix")
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / (matrix.shape[0] - 1)
    covariance = (covariance + covariance.T) * 0.5
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = float(eigenvalues.sum())
    if not math.isfinite(total) or total <= 0:
        raise ValueError(f"{label}: non-positive covariance trace {total}")
    probabilities = eigenvalues[eigenvalues > 0] / total
    effective_rank = float(np.exp(-(probabilities * np.log(probabilities)).sum()))
    mean_channel_variance = float(np.diag(covariance).mean())
    return {
        "sample_count": int(matrix.shape[0]),
        "dimension": int(matrix.shape[1]),
        "mean_channel_variance": finite_float(mean_channel_variance, f"{label} variance"),
        "covariance_effective_rank": finite_float(effective_rank, f"{label} rank"),
        "covariance_trace": total,
        "covariance_eigenvalues_sha256": array_sha256(eigenvalues.astype(np.float64)),
    }


def fit_pca_visible_train(
    raw_train: np.ndarray,
    visible_mask: np.ndarray,
    dimension: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    fit_matrix = np.asarray(raw_train[:, visible_mask], dtype=np.float64).reshape(
        -1, raw_train.shape[-1]
    )
    if dimension <= 0 or dimension > min(fit_matrix.shape):
        raise ValueError(
            f"PCA dimension {dimension} exceeds fit matrix limits {fit_matrix.shape}"
        )
    mean = fit_matrix.mean(axis=0)
    centered = fit_matrix - mean
    covariance = centered.T @ centered / (fit_matrix.shape[0] - 1)
    covariance = (covariance + covariance.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    components = eigenvectors[:, order[:dimension]].T
    # Resolve eigenvector sign ambiguity for deterministic hashes and projections.
    for index in range(len(components)):
        pivot = int(np.argmax(np.abs(components[index])))
        if components[index, pivot] < 0:
            components[index] *= -1
    total_variance = float(eigenvalues.sum())
    retained_ratio = float(eigenvalues[:dimension].sum() / total_variance)
    metadata = {
        "algorithm": "symmetric covariance eigendecomposition",
        "fit_split": "clean_train",
        "fit_scope": "visible_positions_outside_fixed_blackout_only",
        "whiten": False,
        "input_dimension": int(fit_matrix.shape[1]),
        "output_dimension": dimension,
        "fit_samples": int(fit_matrix.shape[0]),
        "mean_sha256": array_sha256(mean.astype(np.float64)),
        "components_sha256": array_sha256(components.astype(np.float64)),
        "eigenvalues_sha256": array_sha256(eigenvalues.astype(np.float64)),
        "retained_explained_variance_ratio": retained_ratio,
        "top_eigenvalues": [float(value) for value in eigenvalues[:dimension]],
    }
    return mean, components, eigenvalues, metadata


def project_features(
    raw: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    flat = np.asarray(raw, dtype=np.float64).reshape(-1, raw.shape[-1])
    projected = (flat - mean) @ components.T
    if not np.isfinite(projected).all():
        raise ValueError("PCA projection produced non-finite values")
    return projected.astype(np.float32).reshape(*raw.shape[:-1], components.shape[0])


def baseline_quality(
    clean_val: np.ndarray,
    constant_target: np.ndarray,
    target_valid_mask: np.ndarray,
    occ_start: int,
    occ_end: int,
) -> Dict[str, float]:
    valid = np.asarray(target_valid_mask, dtype=np.bool_)
    expected_valid = np.ones_like(valid)
    expected_valid[occ_start:occ_end] = False
    if not np.array_equal(valid, expected_valid):
        raise ValueError("baseline target-valid mask does not match blackout interval")
    constant_error = float(
        np.mean((clean_val - constant_target.reshape(1, 1, -1)) ** 2)
    )
    persistence_error = float(
        np.mean((clean_val[:, 1:] - clean_val[:, :-1]) ** 2)
    )
    hold = clean_val[:, occ_start - 1 : occ_start]
    hold_error = float(np.mean((clean_val[:, occ_start:occ_end] - hold) ** 2))
    values = {
        "constant_train_mean_mse": constant_error,
        "immediate_persistence_mse": persistence_error,
        "last_visible_hold_mse": hold_error,
    }
    for name, value in values.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"quality baseline {name} must be finite and positive, got {value}")
    return values


def split_arrays(
    split: SplitPixels,
    clean_features: np.ndarray,
    input_features: np.ndarray,
    target_valid_mask: np.ndarray,
    constant_target: np.ndarray,
    feature_dim: int,
    manifest_sha256: str,
    clean_env: str,
) -> Dict[str, np.ndarray]:
    arrays = {
        "schema_version": np.asarray(FEATURE_SCHEMA_VERSION, dtype=np.int64),
        "split": np.asarray(split.split),
        "clean_env": np.asarray(clean_env),
        "occ_env": np.asarray(f"{clean_env}.occ"),
        "features_input": np.asarray(input_features, dtype=np.float32),
        "features_target": np.asarray(clean_features, dtype=np.float32),
        "actions": np.asarray(split.actions, dtype=np.int64),
        "target_valid_mask": np.asarray(target_valid_mask, dtype=np.bool_),
        "n_actions": np.asarray(split.n_actions, dtype=np.int64),
        "constant_target": np.asarray(constant_target, dtype=np.float32),
        "feature_dim": np.asarray(feature_dim, dtype=np.int64),
        "manifest_sha256": np.asarray(manifest_sha256),
    }
    if set(arrays) != NPZ_FIELDS:
        raise AssertionError(f"internal NPZ schema mismatch: {sorted(arrays)}")
    return arrays


def content_hashes(arrays: Mapping[str, np.ndarray]) -> Dict[str, str]:
    return {name: array_sha256(np.asarray(arrays[name])) for name in sorted(NPZ_CONTENT_FIELDS)}


def atomic_write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        with temporary.open("w") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_npz_artifact(
    path: Path,
    split: str,
    manifest_hash: str,
    expected_content_hashes: Mapping[str, str],
    clean_env: str,
    feature_dim: int,
    n_actions: int,
    episodes: int,
    length: int,
    expected_constant_hash: str,
) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"resume artifact missing/empty: {path}")
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != NPZ_FIELDS:
            raise ValueError(f"{path}: NPZ fields={sorted(data.files)}, expected={sorted(NPZ_FIELDS)}")
        scalar_checks = {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "split": split,
            "clean_env": clean_env,
            "occ_env": f"{clean_env}.occ",
            "n_actions": n_actions,
            "feature_dim": feature_dim,
            "manifest_sha256": manifest_hash,
        }
        for name, expected in scalar_checks.items():
            actual = scalar_value(data[name], f"{path}:{name}")
            if actual != expected:
                raise ValueError(f"{path}: {name}={actual!r}, expected {expected!r}")
        if np.asarray(data["features_input"]).shape != (episodes, length, feature_dim):
            raise ValueError(f"{path}: invalid features_input shape")
        if np.asarray(data["features_target"]).shape != (episodes, length, feature_dim):
            raise ValueError(f"{path}: invalid features_target shape")
        if np.asarray(data["actions"]).shape != (episodes, length - 1):
            raise ValueError(f"{path}: invalid actions shape")
        if np.asarray(data["target_valid_mask"]).shape != (length,):
            raise ValueError(f"{path}: invalid target_valid_mask shape")
        if np.asarray(data["constant_target"]).shape != (feature_dim,):
            raise ValueError(f"{path}: invalid constant_target shape")
        exact_dtypes = {
            "schema_version": np.dtype(np.int64),
            "features_input": np.dtype(np.float32),
            "features_target": np.dtype(np.float32),
            "actions": np.dtype(np.int64),
            "target_valid_mask": np.dtype(np.bool_),
            "n_actions": np.dtype(np.int64),
            "constant_target": np.dtype(np.float32),
            "feature_dim": np.dtype(np.int64),
        }
        for name, expected_dtype in exact_dtypes.items():
            actual_dtype = np.asarray(data[name]).dtype
            if actual_dtype != expected_dtype:
                raise ValueError(
                    f"{path}: {name} dtype={actual_dtype}, expected {expected_dtype}"
                )
        actions = np.asarray(data["actions"])
        if actions.size == 0 or actions.min() < 0 or actions.max() >= n_actions:
            raise ValueError(f"{path}: invalid action indices/count")
        expected_mask = np.ones(length, dtype=np.bool_)
        occ_start = length // 3
        occ_end = min(length, occ_start + max(4, length // 5))
        expected_mask[occ_start:occ_end] = False
        if not np.array_equal(np.asarray(data["target_valid_mask"]), expected_mask):
            raise ValueError(f"{path}: target_valid_mask violates blackout protocol")
        if not np.array_equal(
            np.asarray(data["features_input"])[:, expected_mask],
            np.asarray(data["features_target"])[:, expected_mask],
        ):
            raise ValueError(f"{path}: input/target differ outside blackout")
        if array_sha256(np.asarray(data["constant_target"])) != expected_constant_hash:
            raise ValueError(f"{path}: constant_target does not match manifest")
        if set(expected_content_hashes) != NPZ_CONTENT_FIELDS:
            raise ValueError(f"{path}: manifest content-hash field set is incomplete")
        for name, expected_hash in expected_content_hashes.items():
            actual_hash = array_sha256(np.asarray(data[name]))
            if actual_hash != expected_hash:
                raise ValueError(f"{path}: content hash mismatch for {name}")


def try_strict_resume(
    manifest_path: Path,
    train_path: Path,
    val_path: Path,
    expected_runtime: Mapping[str, Any],
) -> bool:
    existence = (manifest_path.exists(), train_path.exists(), val_path.exists())
    if existence == (False, False, False):
        return False
    if existence != (True, True, True):
        raise RuntimeError(
            f"partial precompute artifacts exist; refusing overwrite: "
            f"manifest/train/val={existence}"
        )
    manifest_bytes = manifest_path.read_bytes()
    manifest_hash = sha256_bytes(manifest_bytes)
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, Mapping):
        raise TypeError(f"{manifest_path}: manifest is not an object")
    artifact_files = manifest.get("artifact_files")
    if not isinstance(artifact_files, Mapping):
        raise ValueError("resume manifest lacks artifact_files")
    if artifact_files.get("train") != train_path.name:
        raise ValueError("resume manifest train artifact name mismatch")
    if artifact_files.get("val") != val_path.name:
        raise ValueError("resume manifest val artifact name mismatch")
    for key, expected in expected_runtime.items():
        actual = manifest.get(key)
        # Artifacts produced before this explicit field was added implement protocol 1.
        if key == 'producer_protocol_version' and actual is None:
            actual = 1
        if actual != expected:
            raise ValueError(f"resume provenance mismatch for {key}")
    output_content = manifest.get("output_content_hashes")
    if not isinstance(output_content, Mapping):
        raise ValueError("resume manifest lacks output_content_hashes")
    config = manifest.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("resume manifest lacks config")
    constant_hash = str(manifest.get("constant_target_sha256", ""))
    validate_npz_artifact(
        train_path,
        "train",
        manifest_hash,
        output_content["train"],
        str(config["clean_env"]),
        int(config["feature_dim"]),
        int(config["n_actions"]),
        int(config["train_episodes"]),
        int(config["length"]),
        constant_hash,
    )
    validate_npz_artifact(
        val_path,
        "val",
        manifest_hash,
        output_content["val"],
        str(config["clean_env"]),
        int(config["feature_dim"]),
        int(config["n_actions"]),
        int(config["val_episodes"]),
        int(config["length"]),
        constant_hash,
    )
    print(f"strict resume validated: {manifest_path}")
    return True


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute paired schema-v1 DINOv2/PCA features from clean robot pixels."
    )
    parser.add_argument("--clean-env", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("outputs/popgym_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dino_clean_features"))
    parser.add_argument("--train-episodes", type=int, default=600)
    parser.add_argument("--val-episodes", type=int, default=150)
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--min-mean-raw-variance", type=float, default=1e-4)
    parser.add_argument("--min-effective-rank", type=float, default=2.0)
    parser.add_argument("--min-retained-variance-ratio", type=float, default=0.95)
    args = parser.parse_args(argv)
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.clean_env.endswith(".occ"):
        parser.error("--clean-env must name the clean environment, not a .occ variant")
    if not args.clean_env.startswith(("dmc:", "ogbench:")):
        parser.error("--clean-env must be a dmc: or ogbench: environment")
    for field in ("train_episodes", "val_episodes", "length", "batch_size", "dim"):
        if getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.length < 6:
        parser.error("--length must be at least 6 for the fixed blackout protocol")
    if args.dim > RAW_FEATURE_DIM:
        parser.error(f"--dim cannot exceed the raw DINO dimension ({RAW_FEATURE_DIM})")
    for field in (
        "min_mean_raw_variance",
        "min_effective_rank",
        "min_retained_variance_ratio",
    ):
        value = getattr(args, field)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{field.replace('_', '-')} must be finite and positive")
    if args.min_retained_variance_ratio > 1:
        parser.error("--min-retained-variance-ratio cannot exceed 1")
    return args


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested ({value}) but unavailable")
    return device


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.device)
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args.output_dir.mkdir(parents=True, exist_ok=True)
    safe = safe_env_name(args.clean_env)
    train_path = args.output_dir / f"{safe}_train.npz"
    val_path = args.output_dir / f"{safe}_val.npz"
    manifest_path = args.output_dir / f"{safe}_manifest.json"

    train_pixels = load_split_pixels(
        args.clean_env,
        args.data_dir,
        args.train_episodes,
        args.length,
        TRAIN_ROLLOUT_SEED,
        "train",
    )
    val_pixels = load_split_pixels(
        args.clean_env,
        args.data_dir,
        args.val_episodes,
        args.length,
        VAL_ROLLOUT_SEED,
        "val",
    )
    if train_pixels.n_actions != val_pixels.n_actions:
        raise ValueError("train/val n_actions mismatch")
    if not np.array_equal(train_pixels.action_prototypes, val_pixels.action_prototypes):
        raise ValueError("train/val action prototypes differ")
    sources = {**train_pixels.sources, **val_pixels.sources}

    model, model_metadata = load_dino(device)
    import timm

    config = {
        "clean_env": args.clean_env,
        "occ_env": f"{args.clean_env}.occ",
        "data_dir": str(args.data_dir),
        "train_episodes": args.train_episodes,
        "val_episodes": args.val_episodes,
        "length": args.length,
        "img_size": IMG_SIZE,
        "train_rollout_seed": TRAIN_ROLLOUT_SEED,
        "val_rollout_seed": VAL_ROLLOUT_SEED,
        "prototype_seed": PROTOTYPE_SEED,
        "pixel_schema_version": PIXEL_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_dim": args.dim,
        "raw_feature_dim": RAW_FEATURE_DIM,
        "n_actions": train_pixels.n_actions,
        "device": str(device),
        "batch_size": args.batch_size,
        "quality_thresholds": {
            "min_mean_raw_channel_variance": args.min_mean_raw_variance,
            "min_covariance_effective_rank": args.min_effective_rank,
            "min_retained_explained_variance_ratio": args.min_retained_variance_ratio,
        },
    }
    software = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "timm": timm.__version__,
    }
    expected_runtime = {
        "producer_protocol_version": PRODUCER_PROTOCOL_VERSION,
        "config": config,
        "source_pixel_caches": sources,
        "dino_model": model_metadata,
        "software": software,
    }
    if try_strict_resume(
        manifest_path,
        train_path,
        val_path,
        expected_runtime,
    ):
        return 0

    print(
        f"encoding clean frames only: train={train_pixels.clean_obs.shape}, "
        f"val={val_pixels.clean_obs.shape}, device={device}"
    )
    raw_train = encode_clean_frames(
        model,
        train_pixels.clean_obs,
        device,
        args.batch_size,
    )
    raw_val = encode_clean_frames(
        model,
        val_pixels.clean_obs,
        device,
        args.batch_size,
    )
    raw_black = encode_black_frame(model, device)
    occ_start = args.length // 3
    occ_end = min(args.length, occ_start + max(4, args.length // 5))
    target_valid_mask = np.ones(args.length, dtype=np.bool_)
    target_valid_mask[occ_start:occ_end] = False

    visible_train = raw_train[:, target_valid_mask].reshape(-1, RAW_FEATURE_DIM)
    raw_quality = covariance_quality(visible_train, "raw clean-train visible features")
    if raw_quality["mean_channel_variance"] < args.min_mean_raw_variance:
        raise ValueError(
            f"raw feature variance {raw_quality['mean_channel_variance']:.6g} < "
            f"{args.min_mean_raw_variance}"
        )
    if raw_quality["covariance_effective_rank"] < args.min_effective_rank:
        raise ValueError(
            f"raw effective rank {raw_quality['covariance_effective_rank']:.6g} < "
            f"{args.min_effective_rank}"
        )

    pca_mean, pca_components, _eigenvalues, pca_metadata = fit_pca_visible_train(
        raw_train,
        target_valid_mask,
        args.dim,
    )
    retained = pca_metadata["retained_explained_variance_ratio"]
    if retained < args.min_retained_variance_ratio:
        raise ValueError(
            f"PCA retained variance ratio {retained:.6g} < "
            f"{args.min_retained_variance_ratio}"
        )

    raw_train_input = np.array(raw_train, copy=True)
    raw_val_input = np.array(raw_val, copy=True)
    raw_train_input[:, occ_start:occ_end] = raw_black
    raw_val_input[:, occ_start:occ_end] = raw_black
    if not np.array_equal(raw_train_input[:, :occ_start], raw_train[:, :occ_start]):
        raise AssertionError("derived train raw features changed before blackout")
    if not np.array_equal(raw_train_input[:, occ_end:], raw_train[:, occ_end:]):
        raise AssertionError("derived train raw features changed after blackout")
    if not np.array_equal(raw_val_input[:, :occ_start], raw_val[:, :occ_start]):
        raise AssertionError("derived val raw features changed before blackout")
    if not np.array_equal(raw_val_input[:, occ_end:], raw_val[:, occ_end:]):
        raise AssertionError("derived val raw features changed after blackout")
    if not np.array_equal(
        raw_train_input[:, occ_start:occ_end],
        np.broadcast_to(raw_black, raw_train_input[:, occ_start:occ_end].shape),
    ):
        raise AssertionError("derived train blackout features do not equal black feature")
    if not np.array_equal(
        raw_val_input[:, occ_start:occ_end],
        np.broadcast_to(raw_black, raw_val_input[:, occ_start:occ_end].shape),
    ):
        raise AssertionError("derived val blackout features do not equal black feature")

    train_target = project_features(raw_train, pca_mean, pca_components)
    val_target = project_features(raw_val, pca_mean, pca_components)
    projected_black = project_features(
        raw_black.reshape(1, 1, -1), pca_mean, pca_components
    )[0, 0]
    # Construct the projected occluded streams from the projected clean streams.
    # This is algebraically identical to projecting ``raw_*_input`` and guarantees
    # the consumer's exact-equality invariant outside the blackout interval.
    train_input = np.array(train_target, copy=True)
    val_input = np.array(val_target, copy=True)
    train_input[:, occ_start:occ_end] = projected_black
    val_input[:, occ_start:occ_end] = projected_black
    # A strict POMDP baseline may use only positions visible in the training stream.
    constant_target = train_target[:, target_valid_mask].mean(axis=(0, 1), dtype=np.float64)
    constant_target = constant_target.astype(np.float32)
    if not np.allclose(
        train_input[:, occ_start:occ_end],
        np.broadcast_to(projected_black, train_input[:, occ_start:occ_end].shape),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise AssertionError("projected train blackout does not equal projected black feature")
    if not np.allclose(
        val_input[:, occ_start:occ_end],
        np.broadcast_to(projected_black, val_input[:, occ_start:occ_end].shape),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise AssertionError("projected val blackout does not equal projected black feature")

    projected_train_visible = train_target[:, target_valid_mask].reshape(-1, args.dim)
    projected_train_quality = covariance_quality(
        projected_train_visible,
        "projected clean-train visible features",
    )
    projected_val_visible = val_target[:, target_valid_mask].reshape(-1, args.dim)
    projected_val_quality = covariance_quality(
        projected_val_visible,
        "projected clean-val valid-target features",
    )
    baselines = baseline_quality(
        val_target,
        constant_target,
        target_valid_mask,
        occ_start,
        occ_end,
    )

    # Build manifest content hashes without manifest_sha256.  The exact manifest-file
    # hash is checked separately as the one excluded NPZ scalar at resume time.
    placeholder_hash = "0" * 64
    train_arrays_placeholder = split_arrays(
        train_pixels,
        train_target,
        train_input,
        target_valid_mask,
        constant_target,
        args.dim,
        placeholder_hash,
        args.clean_env,
    )
    val_arrays_placeholder = split_arrays(
        val_pixels,
        val_target,
        val_input,
        target_valid_mask,
        constant_target,
        args.dim,
        placeholder_hash,
        args.clean_env,
    )
    output_content_hashes = {
        "train": content_hashes(train_arrays_placeholder),
        "val": content_hashes(val_arrays_placeholder),
    }
    core: Dict[str, Any] = {
        **expected_runtime,
        "feature_semantics": {
            "features_target": "PCA projection of clean DINO features",
            "features_input": (
                "same clean features with fixed blackout positions replaced by the "
                "exact-preprocessing all-black-frame DINO feature"
            ),
            "target_valid_mask": "true outside blackout; false inside blackout",
            "constant_target": "mean of projected visible clean-train target frames only",
            "occ_start": occ_start,
            "occ_end": occ_end,
            "clean_val_baseline_scopes": {
                "constant_train_mean_mse": "all clean validation target frames",
                "immediate_persistence_mse": "all adjacent clean validation target frames",
                "last_visible_hold_mse": "clean validation blackout target frames",
            },
        },
        "raw_quality_clean_train_visible": raw_quality,
        "pca": pca_metadata,
        "pca_parameters": {
            "mean_float64": pca_mean.tolist(),
            "components_float64": pca_components.tolist(),
        },
        "projected_quality_clean_train_visible": projected_train_quality,
        "projected_quality_clean_val_valid": projected_val_quality,
        "clean_val_baselines": baselines,
        "dino_feature_hashes": {
            "clean_train_raw_sha256": array_sha256(raw_train),
            "clean_val_raw_sha256": array_sha256(raw_val),
            "derived_occ_train_raw_sha256": array_sha256(raw_train_input),
            "derived_occ_val_raw_sha256": array_sha256(raw_val_input),
        },
        "black_feature": {
            "raw_sha256": array_sha256(raw_black.astype(np.float32)),
            "projected_sha256": array_sha256(projected_black.astype(np.float32)),
            "preprocessing_matches_clean_encoding": True,
        },
        "constant_target_sha256": array_sha256(constant_target),
        "output_content_hashes": output_content_hashes,
    }
    manifest = {
        **core,
        "artifact_files": {
            "train": train_path.name,
            "val": val_path.name,
        },
    }
    # The training consumer compares the scalar embedded in each NPZ against the
    # SHA-256 of the exact manifest file bytes.  Write the immutable manifest first,
    # hash those bytes, then write both NPZs.  NPZ semantic content hashes deliberately
    # exclude only that scalar, which avoids an impossible circular file-hash graph.
    atomic_write_json(manifest_path, manifest)
    manifest_hash = file_sha256(manifest_path)
    train_arrays = split_arrays(
        train_pixels,
        train_target,
        train_input,
        target_valid_mask,
        constant_target,
        args.dim,
        manifest_hash,
        args.clean_env,
    )
    val_arrays = split_arrays(
        val_pixels,
        val_target,
        val_input,
        target_valid_mask,
        constant_target,
        args.dim,
        manifest_hash,
        args.clean_env,
    )

    atomic_write_npz(train_path, train_arrays)
    atomic_write_npz(val_path, val_arrays)
    # Validate the committed files through the same strict resume path.
    if not try_strict_resume(manifest_path, train_path, val_path, expected_runtime):
        raise AssertionError("committed artifact validation unexpectedly returned false")
    print(
        f"wrote {train_path}\n"
        f"wrote {val_path}\n"
        f"wrote {manifest_path}\n"
        f"raw variance={raw_quality['mean_channel_variance']:.6g}, "
        f"raw rank={raw_quality['covariance_effective_rank']:.3f}, "
        f"PCA retained={retained:.3%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
