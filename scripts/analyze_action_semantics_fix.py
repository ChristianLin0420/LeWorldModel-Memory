"""Re-evaluate the canonical robotic checkpoints with corrected action semantics.

The historical training trajectories used rollout seed 0, so action indices during
training already referred to the seed-0 continuous-action prototypes.  Historical
validation trajectories incorrectly regenerated the prototypes with rollout seed
7777.  Retraining is therefore unnecessary: this script evaluates the saved weights
on schema-v3 validation trajectories whose rollout seed remains 7777 while their
prototype seed is fixed to 0. Occluded views are exact masked copies of clean caches.

The evaluator is deliberately strict:

* ``outputs/robotic`` must contain exactly 60 canonical checkpoints
  (48 none/multi full+occluded DMC runs and 12 occluded SMT runs).
* ``outputs/ogbench`` must contain exactly 15 canonical checkpoints
  (12 none/multi full+occluded runs and 3 occluded SMT runs).
* Every checkpoint and historical metrics file must be complete, finite, and match
  the expected training configuration.
* Aggregation requires exactly 75 fresh per-run re-evaluation records.

All reported MSEs are *self-latent* errors: each model predicts targets from its own
encoder coordinate system.  For ``.occ`` environments, the middle-window targets are
latents of black/occluded frames, not clean unoccluded targets.  These values and their
cross-model deltas are descriptive; they are not a clean common-target comparison.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.data import PopgymDataset  # noqa: E402
from lewm.models.memory_model import MemoryLeWorldModel  # noqa: E402


DMC_TASKS = (
    "reacher.hard",
    "ball_in_cup.catch",
    "finger.spin",
    "cheetah.run",
)
DMC_ENVS = tuple(
    env
    for task in DMC_TASKS
    for env in (f"dmc:{task}", f"dmc:{task}.occ")
)
OGBENCH_ENVS = ("ogbench:cube-single", "ogbench:cube-single.occ")
ALL_ENVS = DMC_ENVS + OGBENCH_ENVS
SEEDS = (0, 1, 2)
BASE_DESIGNS = ("none", "multi")

VAL_EPISODES = 150
ROLLOUT_SEED = 7777
PROTOTYPE_SEED = 0
DATASET_SCHEMA_VERSION = 3
LENGTH = 32
IMG_SIZE = 64
N_ACTIONS = 6

COMMON_CHECKPOINT_CONFIG: Mapping[str, Any] = {
    "num_episodes": 600,
    "val_episodes": VAL_EPISODES,
    "length": LENGTH,
    "img_size": IMG_SIZE,
    "epochs": 30,
    "batch_size": 64,
    "lr": 3e-4,
    "weight_decay": 1e-5,
    "num_workers": 2,
    "no_amp": False,
    "patch_size": 8,
    "embed_dim": 128,
    "encoder_layers": 6,
    "encoder_heads": 4,
    "predictor_layers": 4,
    "predictor_heads": 8,
    "history_len": 3,
    "dropout": 0.1,
    "sigreg_lambda": 0.1,
    "sigreg_projections": 512,
    "tau_fast": 3.0,
    "tau_slow": 25.0,
    "fixed_alpha": True,
    "device": "cuda",
}

REEVAL_METRICS = (
    "self_latent_mse_all",
    "self_latent_mse_pre_window",
    "self_latent_mse_mid_window",
    "self_latent_mse_post_window",
    "infl_fast",
    "infl_slow",
)

PER_RUN_FIELDS = (
    "run",
    "family",
    "env",
    "task",
    "variant",
    "design",
    "seed",
    "source_checkpoint",
    "checkpoint_sha256",
    "checkpoint_size",
    "checkpoint_mtime_ns",
    "training_config_fingerprint",
    "checkpoint_prototype_seed",
    "checkpoint_prototype_seed_source",
    "eval_dataset_schema_version",
    "eval_prototype_seed",
    "eval_action_prototype_hash",
    "eval_cache_path",
    "eval_cache_sha256",
    "eval_cache_size",
    "eval_cache_mtime_ns",
    "eval_rollout_seed",
    "eval_episodes",
    "length",
    "img_size",
    "history_len",
    "occ_start",
    "occ_end",
    "mid_target_kind",
    "analysis_device",
    "analysis_batch_size",
    "self_latent_mse_all",
    "self_latent_mse_pre_window",
    "self_latent_mse_mid_window",
    "self_latent_mse_post_window",
    "infl_fast",
    "infl_slow",
)

GROUPED_FIELDS = (
    "family",
    "env",
    "task",
    "variant",
    "design",
    "n_seeds",
    "seed_list",
    "eval_dataset_schema_version",
    "eval_prototype_seed",
    "eval_action_prototype_hash",
    "eval_rollout_seed",
    "mid_target_kind",
    "training_config_fingerprint",
) + tuple(
    field
    for metric in REEVAL_METRICS
    for field in (f"{metric}_mean", f"{metric}_std")
)

PAIRED_METRICS = (
    "self_latent_mse_all",
    "self_latent_mse_pre_window",
    "self_latent_mse_mid_window",
    "self_latent_mse_post_window",
    "infl_fast",
    "infl_slow",
)

PAIRED_FIELDS = (
    "family",
    "env",
    "task",
    "variant",
    "seed",
    "comparison",
    "reference_design",
    "comparison_design",
    "eval_prototype_seed",
    "eval_rollout_seed",
    "mid_target_kind",
) + tuple(
    field
    for metric in PAIRED_METRICS
    for field in (
        f"reference_{metric}",
        f"comparison_{metric}",
        f"delta_{metric}",
    )
) + (
    "relative_delta_self_latent_mse_all",
    "relative_delta_self_latent_mse_post_window",
)


@dataclass(frozen=True)
class RunSpec:
    family: str
    env: str
    design: str
    seed: int

    @property
    def run(self) -> str:
        return f"lewm-{self.env}-{self.design}-s{self.seed}"

    @property
    def variant(self) -> str:
        return "occluded" if self.env.endswith(".occ") else "full"

    @property
    def task(self) -> str:
        return self.env[:-4] if self.env.endswith(".occ") else self.env


def expected_specs() -> List[RunSpec]:
    specs: List[RunSpec] = []
    for family, envs in (("dmc", DMC_ENVS), ("ogbench", OGBENCH_ENVS)):
        for env in envs:
            designs = BASE_DESIGNS + (("smt",) if env.endswith(".occ") else ())
            for design in designs:
                for seed in SEEDS:
                    specs.append(RunSpec(family, env, design, seed))
    if len(specs) != 75:
        raise AssertionError(f"internal expected-factorial error: {len(specs)} runs")
    return specs


def source_root(spec: RunSpec, robotic_root: Path, ogbench_root: Path) -> Path:
    return robotic_root if spec.family == "dmc" else ogbench_root


def as_args_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise TypeError(f"checkpoint args must be a mapping or Namespace, got {type(value).__name__}")


def finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} is non-finite: {value!r}")
    return result


def values_match(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return actual == expected


def canonical_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite configuration value: {value!r}")
        return value
    if isinstance(value, np.generic):
        return canonical_json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): canonical_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [canonical_json_value(item) for item in value]
    raise TypeError(f"unsupported configuration value {value!r} ({type(value).__name__})")


def config_fingerprint(config: Mapping[str, Any]) -> str:
    payload = json.dumps(
        canonical_json_value(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def expected_family_identity(spec: RunSpec) -> Mapping[str, str]:
    if spec.family == "dmc":
        project = "lewm-memory-robotic"
        tag = "exp:robotic_smt" if spec.design == "smt" else "exp:robotic"
        output_dir = "outputs/robotic"
    else:
        project = "lewm-memory-ogbench"
        tag = "exp:ogbench_smt" if spec.design == "smt" else "exp:ogbench"
        output_dir = "outputs/ogbench"
    return {"wandb_project": project, "extra_tag": tag, "output_dir": output_dir}


def validate_checkpoint_config(args: Mapping[str, Any], spec: RunSpec) -> Tuple[str, int, str]:
    identity = {
        "env_id": spec.env,
        "memory_mode": spec.design,
        "seed": spec.seed,
        **expected_family_identity(spec),
    }
    for key, expected in {**COMMON_CHECKPOINT_CONFIG, **identity}.items():
        if key not in args:
            raise ValueError(f"{spec.run}: checkpoint args missing {key!r}")
        if not values_match(args[key], expected):
            raise ValueError(
                f"{spec.run}: checkpoint config {key}={args[key]!r}, expected {expected!r}"
            )

    if spec.design == "smt" and args.get("smt_router") != "sigmoid":
        raise ValueError(
            f"{spec.run}: SMT router={args.get('smt_router')!r}, expected 'sigmoid'"
        )

    if "prototype_seed" in args:
        checkpoint_prototype_seed = int(args["prototype_seed"])
        prototype_source = "checkpoint"
    else:
        # Historical runner behavior: prototypes were drawn from the rollout RNG.
        # Training rollout seed was fixed at 0, hence its prototypes were seed 0.
        checkpoint_prototype_seed = 0
        prototype_source = "legacy_train_rollout_seed_0"
    if checkpoint_prototype_seed != PROTOTYPE_SEED:
        raise ValueError(
            f"{spec.run}: checkpoint prototype seed {checkpoint_prototype_seed}, "
            f"expected {PROTOTYPE_SEED}"
        )

    if "data_dir" in args and args["data_dir"] != "outputs/popgym_data":
        raise ValueError(
            f"{spec.run}: data_dir={args['data_dir']!r}, expected 'outputs/popgym_data'"
        )

    outcome_config = {
        key: args[key]
        for key in COMMON_CHECKPOINT_CONFIG
    }
    outcome_config.update(
        {
            "prototype_seed": checkpoint_prototype_seed,
            "data_dir": args.get("data_dir", "outputs/popgym_data"),
            "smt_router": args.get("smt_router", "softmax"),
            "design": spec.design,
        }
    )
    return config_fingerprint(outcome_config), checkpoint_prototype_seed, prototype_source


def validate_metric_mapping(metrics: Mapping[str, Any], spec: RunSpec, label: str) -> None:
    for key, expected in (("env", spec.env), ("design", spec.design), ("n_actions", N_ACTIONS)):
        if metrics.get(key) != expected:
            raise ValueError(
                f"{spec.run}: {label} {key}={metrics.get(key)!r}, expected {expected!r}"
            )
    for key in ("val_pred_loss", "infl_fast", "infl_slow"):
        if key not in metrics:
            raise ValueError(f"{spec.run}: {label} missing {key!r}")
        if finite_float(metrics[key], f"{spec.run}: {label}.{key}") < 0:
            raise ValueError(f"{spec.run}: {label}.{key} must be non-negative")
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            finite_float(value, f"{spec.run}: {label}.{key}")


def validate_source_layout(
    specs: Sequence[RunSpec], robotic_root: Path, ogbench_root: Path
) -> None:
    by_family: Dict[str, set[str]] = defaultdict(set)
    for spec in specs:
        by_family[spec.family].add(spec.run)
    for family, root in (("dmc", robotic_root), ("ogbench", ogbench_root)):
        if not root.is_dir():
            raise FileNotFoundError(f"missing source root: {root}")
        actual = {child.name for child in root.iterdir() if child.is_dir()}
        missing = sorted(by_family[family] - actual)
        unexpected = sorted(actual - by_family[family])
        if missing or unexpected:
            raise ValueError(
                f"{root}: source factorial mismatch; missing={missing}, unexpected={unexpected}"
            )
        for run in sorted(actual):
            for filename in ("model.pt", "metrics.json"):
                path = root / run / filename
                if not path.is_file() or path.stat().st_size <= 0:
                    raise ValueError(f"incomplete source artifact: {path}")


def load_and_validate_source(
    spec: RunSpec, robotic_root: Path, ogbench_root: Path
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    run_dir = source_root(spec, robotic_root, ogbench_root) / spec.run
    checkpoint_path = run_dir / "model.pt"
    metrics_path = run_dir / "metrics.json"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"{checkpoint_path} is not a checkpoint mapping")
    args = as_args_dict(checkpoint.get("args"))
    fingerprint, prototype_seed, prototype_source = validate_checkpoint_config(args, spec)

    with metrics_path.open() as handle:
        historical_metrics = json.load(handle)
    if not isinstance(historical_metrics, Mapping):
        raise TypeError(f"{metrics_path} is not a metric mapping")
    validate_metric_mapping(historical_metrics, spec, "metrics.json")
    final_metrics = checkpoint.get("final_metrics")
    if not isinstance(final_metrics, Mapping):
        raise TypeError(f"{spec.run}: checkpoint final_metrics is not a mapping")
    validate_metric_mapping(final_metrics, spec, "checkpoint.final_metrics")
    if dict(final_metrics) != dict(historical_metrics):
        raise ValueError(f"{spec.run}: metrics.json differs from checkpoint final_metrics")

    stat = checkpoint_path.stat()
    metadata = {
        "args": args,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_size": stat.st_size,
        "checkpoint_mtime_ns": stat.st_mtime_ns,
        "training_config_fingerprint": fingerprint,
        "checkpoint_prototype_seed": prototype_seed,
        "checkpoint_prototype_seed_source": prototype_source,
    }
    return dict(checkpoint), metadata


def corrected_cache_path(data_dir: Path, env: str) -> Path:
    filename = (
        f"{env.replace(':', '_')}_v{DATASET_SCHEMA_VERSION}_proto{PROTOTYPE_SEED}_n{VAL_EPISODES}_"
        f"L{LENGTH}_s{IMG_SIZE}_seed{ROLLOUT_SEED}.npz"
    )
    return data_dir / filename


def validate_corrected_cache(data_dir: Path, env: str) -> Dict[str, Any]:
    path = corrected_cache_path(data_dir, env)
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(
            f"missing corrected validation cache {path}; run run_action_semantics_reeval.sh"
        )
    with np.load(path, allow_pickle=False) as data:
        required = {
            "obs",
            "actions",
            "n_actions",
            "action_prototypes",
            "prototype_seed",
            "schema_version",
            "cache_role",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"{path}: missing cache field(s) {sorted(missing)}")
        if int(data["prototype_seed"]) != PROTOTYPE_SEED:
            raise ValueError(f"{path}: wrong prototype_seed")
        if int(data["schema_version"]) != DATASET_SCHEMA_VERSION:
            raise ValueError(f"{path}: wrong schema_version")
        role = str(data["cache_role"])
        if env.endswith('.occ'):
            if role != 'paired_occluded' or 'clean_env_id' not in data.files:
                raise ValueError(f"{path}: missing paired-clean provenance")
            if str(data['clean_env_id']) != env[:-4]:
                raise ValueError(f"{path}: wrong clean_env_id")
        elif role != 'clean_or_full':
            raise ValueError(f"{path}: wrong cache_role={role!r}")
        if int(data["n_actions"]) != N_ACTIONS:
            raise ValueError(f"{path}: n_actions={int(data['n_actions'])}, expected {N_ACTIONS}")
        prototypes = np.asarray(data["action_prototypes"])
        actions = np.asarray(data["actions"])
        if prototypes.shape[0] != N_ACTIONS or not np.isfinite(prototypes).all():
            raise ValueError(f"{path}: invalid action prototypes shape/values")
        if actions.shape != (VAL_EPISODES, LENGTH - 1):
            raise ValueError(f"{path}: actions shape={actions.shape}, expected {(VAL_EPISODES, LENGTH - 1)}")
        if actions.min() < 0 or actions.max() >= N_ACTIONS:
            raise ValueError(f"{path}: action index outside [0, {N_ACTIONS})")
        prototype_payload = (
            str(prototypes.dtype).encode("ascii")
            + repr(prototypes.shape).encode("ascii")
            + prototypes.tobytes(order="C")
        )
        prototype_hash = hashlib.sha256(prototype_payload).hexdigest()
    stat = path.stat()
    return {
        "eval_action_prototype_hash": prototype_hash,
        "eval_cache_path": str(path.resolve()),
        "eval_cache_sha256": file_sha256(path),
        "eval_cache_size": stat.st_size,
        "eval_cache_mtime_ns": stat.st_mtime_ns,
    }


def validate_all_corrected_caches(data_dir: Path) -> Dict[str, Dict[str, Any]]:
    metadata = {env: validate_corrected_cache(data_dir, env) for env in ALL_ENVS}
    for task in tuple(f"dmc:{name}" for name in DMC_TASKS) + ("ogbench:cube-single",):
        full_hash = metadata[task]["eval_action_prototype_hash"]
        occluded_hash = metadata[f"{task}.occ"]["eval_action_prototype_hash"]
        if full_hash != occluded_hash:
            raise ValueError(
                f"prototype mismatch between full and occluded caches for {task}: "
                f"{full_hash} != {occluded_hash}"
            )
        full_path = corrected_cache_path(data_dir, task)
        occ_path = corrected_cache_path(data_dir, f"{task}.occ")
        with np.load(full_path, allow_pickle=False) as full, np.load(occ_path, allow_pickle=False) as occ:
            full_actions = np.asarray(full['actions'])
            occ_actions = np.asarray(occ['actions'])
            if not np.array_equal(full_actions, occ_actions):
                raise ValueError(f"action-index mismatch between paired caches for {task}")
            full_obs = np.asarray(full['obs'])
            occ_obs = np.asarray(occ['obs'])
            start = LENGTH // 3
            end = min(LENGTH, start + max(4, LENGTH // 5))
            if not np.array_equal(full_obs[:, :start], occ_obs[:, :start]):
                raise ValueError(f"pre-occlusion pixels mismatch for paired caches {task}")
            if not np.array_equal(full_obs[:, end:], occ_obs[:, end:]):
                raise ValueError(f"post-occlusion pixels mismatch for paired caches {task}")
            if np.any(occ_obs[:, start:end]):
                raise ValueError(f"occlusion window is not black for paired cache {task}")
    return metadata


def build_model(args: Mapping[str, Any], action_dim: int) -> MemoryLeWorldModel:
    mode = str(args["memory_mode"])
    impl = mode if mode in ("multi", "gru", "ssm", "retrieval", "smt") else "ema"
    ema_mode = "both" if impl != "ema" else mode
    return MemoryLeWorldModel(
        img_size=int(args["img_size"]),
        patch_size=int(args["patch_size"]),
        embed_dim=int(args["embed_dim"]),
        action_dim=action_dim,
        encoder_layers=int(args["encoder_layers"]),
        encoder_heads=int(args["encoder_heads"]),
        predictor_layers=int(args["predictor_layers"]),
        predictor_heads=int(args["predictor_heads"]),
        history_len=int(args["history_len"]),
        dropout=float(args["dropout"]),
        sigreg_lambda=float(args["sigreg_lambda"]),
        sigreg_projections=int(args["sigreg_projections"]),
        memory_mode=ema_mode,
        memory_impl=impl,
        tau_fast=float(args["tau_fast"]),
        tau_slow=float(args["tau_slow"]),
        learnable_alpha=not bool(args["fixed_alpha"]),
        smt_router=str(args.get("smt_router", "softmax")),
    )


def resolve_device(spec: str) -> torch.device:
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested ({spec}) but unavailable")
    return device


def set_reproducible(seed: int = 4242) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@torch.inference_mode()
def evaluate_model(
    model: MemoryLeWorldModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.to(device).eval()
    history = int(model.history_len)
    times = torch.arange(history, LENGTH, dtype=torch.long)
    occ_start = LENGTH // 3
    occ_end = min(LENGTH, occ_start + max(4, LENGTH // 5))
    phase_masks = {
        "pre_window": times < occ_start,
        "mid_window": (times >= occ_start) & (times < occ_end),
        "post_window": times >= occ_end,
    }
    if any(not bool(mask.any()) for mask in phase_masks.values()):
        raise AssertionError("empty phase mask")

    time_sum = torch.zeros(LENGTH - history, dtype=torch.float64)
    episode_count = 0
    influence_sums = {"infl_fast": 0.0, "infl_slow": 0.0}
    influence_count = 0

    for observations, actions in loader:
        observations = observations.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        z, _m_fast, _m_slow, z_tilde = model.encode_with_memory(observations)
        batch, seq_len, dim = z.shape
        if seq_len != LENGTH:
            raise ValueError(f"evaluation sequence length {seq_len}, expected {LENGTH}")
        windows = seq_len - history
        z_windows = z_tilde.unfold(1, history, 1)[:, :windows]
        z_windows = z_windows.permute(0, 1, 3, 2).reshape(batch * windows, history, dim)
        action_windows = actions.unfold(1, history, 1)[:, :windows]
        action_windows = action_windows.permute(0, 1, 3, 2).reshape(
            batch * windows, history, model.action_dim
        )
        targets = z[:, history:seq_len].reshape(batch * windows, dim)
        predictions = model.predictor(z_windows, action_windows)[:, -1, :]
        errors = ((predictions - targets) ** 2).mean(dim=-1).reshape(batch, windows)
        if not bool(torch.isfinite(errors).all()):
            raise ValueError("non-finite self-latent prediction error")
        time_sum += errors.double().sum(dim=0).cpu()
        episode_count += batch

        influence = model.memory_influence(observations, actions)
        for key in influence_sums:
            values = influence[key]
            if not bool(torch.isfinite(values).all()):
                raise ValueError(f"non-finite {key}")
            influence_sums[key] += float(values.double().sum())
        influence_count += batch

    if episode_count != VAL_EPISODES or influence_count != VAL_EPISODES:
        raise ValueError(
            f"evaluated {episode_count}/{influence_count} episodes, expected {VAL_EPISODES}"
        )
    time_mean = time_sum / episode_count
    metrics = {"self_latent_mse_all": float(time_mean.mean())}
    for phase, mask in phase_masks.items():
        metrics[f"self_latent_mse_{phase}"] = float(time_mean[mask].mean())
    for key, total in influence_sums.items():
        metrics[key] = total / influence_count
    for key, value in metrics.items():
        finite_float(value, key)
        if value < 0:
            raise ValueError(f"negative re-evaluation metric {key}={value}")
    return metrics


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def result_path(output_root: Path, spec: RunSpec) -> Path:
    return output_root / "per_run_json" / f"{spec.run}.json"


def evaluate_shard(
    args: argparse.Namespace,
    specs: Sequence[RunSpec],
    robotic_root: Path,
    ogbench_root: Path,
    output_root: Path,
    data_dir: Path,
) -> None:
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= index < --num-shards")
    device = resolve_device(args.device)
    set_reproducible()
    env_index = {env: index for index, env in enumerate(ALL_ENVS)}
    selected = [
        spec for spec in specs if env_index[spec.env] % args.num_shards == args.shard_index
    ]
    print(
        f"shard {args.shard_index}/{args.num_shards}: {len(selected)} run(s), "
        f"device={device}"
    )
    selected_by_env: Dict[str, List[RunSpec]] = defaultdict(list)
    for spec in selected:
        selected_by_env[spec.env].append(spec)
    cache_metadata = validate_all_corrected_caches(data_dir)

    for env in ALL_ENVS:
        env_specs = selected_by_env.get(env, [])
        if not env_specs:
            continue
        dataset = PopgymDataset(
            env,
            VAL_EPISODES,
            LENGTH,
            IMG_SIZE,
            seed=ROLLOUT_SEED,
            data_dir=str(data_dir),
            prototype_seed=PROTOTYPE_SEED,
        )
        if dataset.n_actions != N_ACTIONS:
            raise ValueError(f"{env}: dataset n_actions={dataset.n_actions}, expected {N_ACTIONS}")
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.data_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.data_workers > 0,
        )
        for spec in env_specs:
            checkpoint, metadata = load_and_validate_source(spec, robotic_root, ogbench_root)
            model = build_model(metadata["args"], dataset.n_actions)
            model.load_state_dict(checkpoint["model_state_dict"])
            metrics = evaluate_model(model, loader, device)
            record: Dict[str, Any] = {
                "schema_version": 1,
                "analysis": "corrected_action_semantics_reevaluation",
                "run": spec.run,
                "family": spec.family,
                "env": spec.env,
                "task": spec.task,
                "variant": spec.variant,
                "design": spec.design,
                "seed": spec.seed,
                "source_checkpoint": metadata["checkpoint_path"],
                "checkpoint_sha256": metadata["checkpoint_sha256"],
                "checkpoint_size": metadata["checkpoint_size"],
                "checkpoint_mtime_ns": metadata["checkpoint_mtime_ns"],
                "training_config_fingerprint": metadata["training_config_fingerprint"],
                "checkpoint_prototype_seed": metadata["checkpoint_prototype_seed"],
                "checkpoint_prototype_seed_source": metadata[
                    "checkpoint_prototype_seed_source"
                ],
                "eval_dataset_schema_version": DATASET_SCHEMA_VERSION,
                "eval_prototype_seed": PROTOTYPE_SEED,
                **cache_metadata[env],
                "eval_rollout_seed": ROLLOUT_SEED,
                "eval_episodes": VAL_EPISODES,
                "length": LENGTH,
                "img_size": IMG_SIZE,
                "history_len": int(metadata["args"]["history_len"]),
                "occ_start": LENGTH // 3,
                "occ_end": min(LENGTH, LENGTH // 3 + max(4, LENGTH // 5)),
                "mid_target_kind": (
                    "self_latent_of_black_occluded_frame"
                    if spec.variant == "occluded"
                    else "self_latent_of_fully_observed_frame"
                ),
                "analysis_device": str(device),
                "analysis_batch_size": args.batch_size,
                **metrics,
            }
            atomic_write_json(result_path(output_root, spec), record)
            print(
                f"  {spec.run}: self-MSE={metrics['self_latent_mse_all']:.6f} "
                f"post={metrics['self_latent_mse_post_window']:.6f} "
                f"infl={metrics['infl_slow']:.4f}"
            )
            del checkpoint, model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del loader, dataset


def load_result(
    path: Path,
    spec: RunSpec,
    metadata: Mapping[str, Any],
    cache_metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    with path.open() as handle:
        record = json.load(handle)
    if not isinstance(record, Mapping):
        raise TypeError(f"{path}: result is not a mapping")
    expected = {
        "schema_version": 1,
        "analysis": "corrected_action_semantics_reevaluation",
        "run": spec.run,
        "family": spec.family,
        "env": spec.env,
        "task": spec.task,
        "variant": spec.variant,
        "design": spec.design,
        "seed": spec.seed,
        "source_checkpoint": metadata["checkpoint_path"],
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "checkpoint_size": metadata["checkpoint_size"],
        "checkpoint_mtime_ns": metadata["checkpoint_mtime_ns"],
        "training_config_fingerprint": metadata["training_config_fingerprint"],
        "checkpoint_prototype_seed": metadata["checkpoint_prototype_seed"],
        "eval_dataset_schema_version": DATASET_SCHEMA_VERSION,
        "eval_prototype_seed": PROTOTYPE_SEED,
        **cache_metadata,
        "eval_rollout_seed": ROLLOUT_SEED,
        "eval_episodes": VAL_EPISODES,
        "length": LENGTH,
        "img_size": IMG_SIZE,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(
                f"{path}: {key}={record.get(key)!r}, expected {value!r}"
            )
    for metric in REEVAL_METRICS:
        if metric not in record:
            raise ValueError(f"{path}: missing {metric}")
        if finite_float(record[metric], f"{path}:{metric}") < 0:
            raise ValueError(f"{path}: negative {metric}={record[metric]}")
    return dict(record)


def aggregate_rows(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["env"]), str(record["design"]))].append(record)
    rows: List[Dict[str, Any]] = []
    for (env, design), members in sorted(grouped.items()):
        seeds = sorted(int(member["seed"]) for member in members)
        if seeds != list(SEEDS):
            raise ValueError(f"{env}/{design}: seeds={seeds}, expected {list(SEEDS)}")
        invariant_fields = (
            "family",
            "task",
            "variant",
            "eval_dataset_schema_version",
            "eval_prototype_seed",
            "eval_action_prototype_hash",
            "eval_rollout_seed",
            "mid_target_kind",
            "training_config_fingerprint",
        )
        row: Dict[str, Any] = {"env": env, "design": design}
        for field in invariant_fields:
            values = {member[field] for member in members}
            if len(values) != 1:
                raise ValueError(f"{env}/{design}: mixed {field}: {values}")
            row[field] = next(iter(values))
        row["n_seeds"] = len(seeds)
        row["seed_list"] = json.dumps(seeds, separators=(",", ":"))
        for metric in REEVAL_METRICS:
            values = np.asarray([finite_float(member[metric], metric) for member in members])
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
        rows.append(row)
    if len(rows) != 25:
        raise ValueError(f"grouped cell count={len(rows)}, expected 25")
    return rows


def paired_rows(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_key = {
        (str(record["env"]), int(record["seed"]), str(record["design"])): record
        for record in records
    }
    rows: List[Dict[str, Any]] = []
    for env in ALL_ENVS:
        comparisons = [("none", "multi")]
        if env.endswith(".occ"):
            comparisons.extend((("none", "smt"), ("multi", "smt")))
        for seed in SEEDS:
            for reference_design, comparison_design in comparisons:
                reference = by_key[(env, seed, reference_design)]
                comparison = by_key[(env, seed, comparison_design)]
                row: Dict[str, Any] = {
                    "family": reference["family"],
                    "env": env,
                    "task": reference["task"],
                    "variant": reference["variant"],
                    "seed": seed,
                    "comparison": f"{comparison_design}-minus-{reference_design}",
                    "reference_design": reference_design,
                    "comparison_design": comparison_design,
                    "eval_prototype_seed": PROTOTYPE_SEED,
                    "eval_rollout_seed": ROLLOUT_SEED,
                    "mid_target_kind": reference["mid_target_kind"],
                }
                for metric in PAIRED_METRICS:
                    reference_value = finite_float(reference[metric], metric)
                    comparison_value = finite_float(comparison[metric], metric)
                    row[f"reference_{metric}"] = reference_value
                    row[f"comparison_{metric}"] = comparison_value
                    row[f"delta_{metric}"] = comparison_value - reference_value
                for metric in (
                    "self_latent_mse_all",
                    "self_latent_mse_post_window",
                ):
                    denominator = float(row[f"reference_{metric}"])
                    if denominator <= 0:
                        raise ValueError(
                            f"{env}/seed{seed}/{reference_design}: non-positive {metric}"
                        )
                    row[f"relative_delta_{metric}"] = float(row[f"delta_{metric}"]) / denominator
                rows.append(row)
    if len(rows) != 60:
        raise ValueError(f"paired row count={len(rows)}, expected 60")
    return rows


def print_summary(pairs: Sequence[Mapping[str, Any]]) -> None:
    selected = [
        row
        for row in pairs
        if row["comparison"] == "multi-minus-none"
        or (row["variant"] == "occluded" and row["comparison"] == "smt-minus-multi")
    ]
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[(str(row["env"]), str(row["comparison"]))].append(row)
    print("\nCorrected action-semantics re-evaluation (delta = comparison - reference)")
    print(
        f"{'env':<31} {'comparison':<17} {'dMSE(all)':>12} "
        f"{'dMSE(post)':>12} {'rel(all)':>10} {'dInfl':>10}"
    )
    for (env, comparison), members in sorted(grouped.items()):
        all_delta = np.mean([row["delta_self_latent_mse_all"] for row in members])
        post_delta = np.mean(
            [row["delta_self_latent_mse_post_window"] for row in members]
        )
        relative = np.mean(
            [row["relative_delta_self_latent_mse_all"] for row in members]
        )
        influence = np.mean([row["delta_infl_slow"] for row in members])
        print(
            f"{env:<31} {comparison:<17} {all_delta:>12.6f} "
            f"{post_delta:>12.6f} {relative:>9.1%} {influence:>10.4f}"
        )
    print(
        "\nCAUTION: MSE is model-specific self-latent error. For .occ runs, the mid-window "
        "target is the black/occluded frame latent, not a clean unoccluded target."
    )


def aggregate(
    specs: Sequence[RunSpec],
    robotic_root: Path,
    ogbench_root: Path,
    output_root: Path,
    data_dir: Path,
) -> None:
    result_dir = output_root / "per_run_json"
    if not result_dir.is_dir():
        raise FileNotFoundError(f"missing per-run result directory: {result_dir}")
    expected_names = {f"{spec.run}.json" for spec in specs}
    actual_names = {path.name for path in result_dir.glob("*.json")}
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    if missing or unexpected:
        raise ValueError(
            f"re-evaluation factorial mismatch: missing={missing}, unexpected={unexpected}"
        )

    cache_metadata = validate_all_corrected_caches(data_dir)
    records: List[Dict[str, Any]] = []
    for spec in specs:
        _checkpoint, metadata = load_and_validate_source(spec, robotic_root, ogbench_root)
        records.append(
            load_result(
                result_path(output_root, spec),
                spec,
                metadata,
                cache_metadata[spec.env],
            )
        )
    if len(records) != 75:
        raise AssertionError(f"record count={len(records)}, expected 75")
    records.sort(key=lambda row: (row["family"], row["env"], row["design"], row["seed"]))
    grouped = aggregate_rows(records)
    pairs = paired_rows(records)

    atomic_write_csv(
        output_root / "action_semantics_fix_per_run.csv",
        PER_RUN_FIELDS,
        records,
    )
    atomic_write_csv(
        output_root / "action_semantics_fix_grouped.csv",
        GROUPED_FIELDS,
        grouped,
    )
    atomic_write_csv(
        output_root / "action_semantics_fix_paired_deltas.csv",
        PAIRED_FIELDS,
        pairs,
    )
    print_summary(pairs)
    print(
        f"\nwrote {len(records)} per-run rows, {len(grouped)} grouped rows, "
        f"and {len(pairs)} paired rows under {output_root}"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict corrected-action re-evaluation for DMC and OGBench checkpoints."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--evaluate-shard", action="store_true")
    mode.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--robotic-root", type=Path, default=REPO_ROOT / "outputs" / "robotic")
    parser.add_argument("--ogbench-root", type=Path, default=REPO_ROOT / "outputs" / "ogbench")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "action_semantics_fix",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "popgym_data",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-workers", type=int, default=2)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args(argv)
    for field in ("robotic_root", "ogbench_root", "output_root", "data_dir"):
        setattr(args, field, getattr(args, field).resolve())
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.data_workers < 0:
        parser.error("--data-workers must be non-negative")
    if args.num_shards <= 0:
        parser.error("--num-shards must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    specs = expected_specs()
    validate_source_layout(specs, args.robotic_root, args.ogbench_root)
    if args.validate_only:
        for spec in specs:
            load_and_validate_source(spec, args.robotic_root, args.ogbench_root)
        print(
            "source validation OK: 60 DMC + 15 OGBench checkpoints; "
            "canonical configs and finite historical metadata"
        )
        return 0
    if args.evaluate_shard:
        evaluate_shard(
            args,
            specs,
            args.robotic_root,
            args.ogbench_root,
            args.output_root,
            args.data_dir,
        )
        return 0
    aggregate(specs, args.robotic_root, args.ogbench_root, args.output_root, args.data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
