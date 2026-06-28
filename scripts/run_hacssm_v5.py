#!/usr/bin/env python3
"""Run the locked, staged HACSSM-v5 fixed-feature experiment.

The prospective pilot is 5 environments x 12 designs x seeds 0--2 (180 runs).
The pilot screen is immutable, but the user's explicit all-experiment instruction
always completes seeds 3--4 (120 runs) and the locked 300-run grid.  A failed pilot
therefore remains a prospective NO_GO even if the descriptive five-seed estimate moves.
Resume is permitted only after strict checkpoint, metric, argument, history, feature,
source, and Git validation.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_popgym.py"
ANALYZE_SCRIPT = REPO_ROOT / "scripts" / "analyze_hacssm_v5.py"
FEATURE_ROOT = REPO_ROOT / "outputs" / "smt_v3_shared" / "dino_features_d128"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "hacssm_v5_shared"
LOG_ROOT = REPO_ROOT / "logs" / "hacssm_v5_shared"
DATA_ROOT = REPO_ROOT / "outputs" / "popgym_data"
PROTOCOL_PATH = OUTPUT_ROOT / "protocol.json"
DECISION_PATH = OUTPUT_ROOT / "pilot_decision.json"
FINAL_DECISION_PATH = OUTPUT_ROOT / "decision.json"
MANIFEST_PATH = OUTPUT_ROOT / "hacssm_v5_manifest.json"
MANIFEST_SHA_PATH = OUTPUT_ROOT / "hacssm_v5_manifest.sha256"
LOCK_PATH = OUTPUT_ROOT / ".run_hacssm_v5.lock"

ENVIRONMENTS = (
    ("dmc:reacher.hard.occ", "dmc:reacher.hard"),
    ("dmc:ball_in_cup.catch.occ", "dmc:ball_in_cup.catch"),
    ("dmc:finger.spin.occ", "dmc:finger.spin"),
    ("dmc:cheetah.run.occ", "dmc:cheetah.run"),
    ("ogbench:cube-single.occ", "ogbench:cube-single"),
)
DESIGNS = (
    "none",
    "ssm",
    "hacsmv4",
    "hacsmv4_noaux",
    "hacsmv4_two_noaux",
    "hacssmv5_ssmcontrol",
    "hacssmv5_fixedbeta_noaux",
    "hacssmv5_noaux",
    "hacssmv5_noaction",
    "hacssmv5_static",
    "hacssmv5_single",
    "hacssmv5",
)
PILOT_SEEDS = (0, 1, 2)
COMPLETION_SEEDS = (3, 4)
ALL_SEEDS = PILOT_SEEDS + COMPLETION_SEEDS
V5_DESIGNS = frozenset(design for design in DESIGNS if design.startswith("hacssmv5"))
HIER_DESIGNS = frozenset(
    design for design in DESIGNS if design.startswith(("hacsmv4", "hacssmv5"))
)
NO_AUX_DESIGNS = frozenset(
    {
        "hacsmv4_noaux",
        "hacsmv4_two_noaux",
        "hacssmv5_ssmcontrol",
        "hacssmv5_fixedbeta_noaux",
        "hacssmv5_noaux",
    }
)

# Scientific protocol values.  A change requires a fresh namespace and schema.
COMMON = {
    "train_episodes": 600,
    "val_episodes": 150,
    "length": 32,
    "feature_dim": 128,
    "batch_size": 64,
    "learning_rate": 3e-4,
    "weight_decay": 1e-5,
    "history_len": 3,
    "predictor_norm": "none",
    "first_post_loss_weight": 0.5,
    "epochs": 200,
    "train_dataloader_workers": 2,
    "prototype_seed": 0,
    "train_rollout_seed": 0,
    "val_rollout_seed": 7777,
    "smt_router": "sigmoid",
    "fixed_alpha": True,
    "wandb": False,
}

SOURCE_FILES = (
    Path("scripts/run_hacssm_v5.py"),
    Path("scripts/analyze_hacssm_v5.py"),
    Path("scripts/train_popgym.py"),
    Path("lewm/data.py"),
    Path("lewm/models/encoder.py"),
    Path("lewm/models/leworldmodel.py"),
    Path("lewm/models/memory.py"),
    Path("lewm/models/memory_model.py"),
    Path("lewm/models/sigreg.py"),
)

PILOT_ANALYSIS_FILES = frozenset(
    {
        "pilot_per_run.csv",
        "pilot_grouped.csv",
        "pilot_paired_contrasts.csv",
        "pilot_convergence.csv",
        "pilot_decision.json",
    }
)
FINAL_ANALYSIS_FILES = frozenset(
    {"per_run.csv", "grouped.csv", "paired_contrasts.csv", "convergence.csv", "decision.json"}
)
TOP_LEVEL_OUTPUT_FILES = frozenset(
    {
        PROTOCOL_PATH.name,
        LOCK_PATH.name,
        MANIFEST_PATH.name,
        MANIFEST_SHA_PATH.name,
        *PILOT_ANALYSIS_FILES,
        *FINAL_ANALYSIS_FILES,
    }
)

_ACTIVE_PROCESSES: set[subprocess.Popen[Any]] = set()
_PROCESS_LOCK = threading.Lock()
_PRINT_LOCK = threading.Lock()


@dataclass(frozen=True, order=True)
class Job:
    stage: str
    seed: int
    occ_env: str
    clean_env: str
    design: str

    @property
    def run_name(self) -> str:
        return f"lewm-{self.occ_env}-{self.design}-s{self.seed}"

    @property
    def run_dir(self) -> Path:
        return OUTPUT_ROOT / self.run_name

    @property
    def model_path(self) -> Path:
        return self.run_dir / "model.pt"

    @property
    def metrics_path(self) -> Path:
        return self.run_dir / "metrics.json"

    @property
    def log_path(self) -> Path:
        safe = re.sub(r"[^A-Za-z0-9]+", "_", self.run_name).strip("_")
        return LOG_ROOT / f"{safe}.log"


def make_jobs(stage: str, seeds: Sequence[int]) -> tuple[Job, ...]:
    # Seed-major order is fixed so resume never changes worker ownership.
    return tuple(
        Job(stage, seed, occ, clean, design)
        for seed in seeds
        for occ, clean in ENVIRONMENTS
        for design in DESIGNS
    )


PILOT_JOBS = make_jobs("pilot", PILOT_SEEDS)
COMPLETION_JOBS = make_jobs("completion", COMPLETION_SEEDS)
ALL_JOBS = PILOT_JOBS + COMPLETION_JOBS

assert len(PILOT_JOBS) == 180
assert len(COMPLETION_JOBS) == 120
assert len(ALL_JOBS) == 300
assert len({job.run_name for job in ALL_JOBS}) == 300


class RunnerError(RuntimeError):
    """A locked protocol or artifact invariant was violated."""


def rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RunnerError(f"required nonempty file is missing: {path}")
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def reject_non_rfc_json(token: str) -> None:
    raise ValueError(f"non-RFC JSON constant {token}")


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(), parse_constant=reject_non_rfc_json)
    except (OSError, UnicodeError, ValueError) as exc:
        raise RunnerError(f"invalid JSON at {path}: {exc}") from exc


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    if temporary.exists():
        raise RunnerError(f"refusing to reuse temporary path: {temporary}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    atomic_write_bytes(path, payload)


def stable_equal(left: Any, right: Any) -> bool:
    """Recursive strict equality: booleans never compare equal to integers."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            stable_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            stable_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def assert_finite_tree(value: Any, context: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            assert_finite_tree(child, f"{context}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_finite_tree(child, f"{context}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise RunnerError(f"non-finite value at {context}: {value!r}")


def git_provenance() -> tuple[str, str]:
    def invoke(arguments: Sequence[str]) -> str:
        result = subprocess.run(
            ["git", *arguments], cwd=REPO_ROOT, capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            raise RunnerError(
                f"git {' '.join(arguments)} failed ({result.returncode}): {result.stderr.strip()}"
            )
        return result.stdout

    commit = invoke(("rev-parse", "HEAD")).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RunnerError(f"unexpected Git commit id: {commit!r}")
    porcelain = invoke(("status", "--porcelain", "--untracked-files=all"))
    return commit, porcelain


def safe_env(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def feature_paths(clean_env: str) -> tuple[Path, Path, Path]:
    prefix = FEATURE_ROOT / safe_env(clean_env)
    return (
        Path(f"{prefix}_train.npz"),
        Path(f"{prefix}_val.npz"),
        Path(f"{prefix}_manifest.json"),
    )


def feature_snapshot() -> dict[str, dict[str, Any]]:
    if not FEATURE_ROOT.is_dir():
        raise RunnerError(f"fixed feature root is missing: {FEATURE_ROOT}")
    expected: set[Path] = set()
    records: dict[str, dict[str, Any]] = {}
    for occ_env, clean_env in ENVIRONMENTS:
        train_path, val_path, manifest_path = feature_paths(clean_env)
        expected.update((train_path, val_path, manifest_path))
        manifest = read_json(manifest_path)
        config = manifest.get("config") if isinstance(manifest, dict) else None
        wanted_config = {
            "occ_env": occ_env,
            "clean_env": clean_env,
            "train_episodes": COMMON["train_episodes"],
            "val_episodes": COMMON["val_episodes"],
            "length": COMMON["length"],
            "feature_dim": COMMON["feature_dim"],
            "feature_schema_version": 1,
            "prototype_seed": COMMON["prototype_seed"],
            "train_rollout_seed": COMMON["train_rollout_seed"],
            "val_rollout_seed": COMMON["val_rollout_seed"],
        }
        if not isinstance(config, dict):
            raise RunnerError(f"feature manifest has no config object: {manifest_path}")
        for key, wanted in wanted_config.items():
            if not stable_equal(config.get(key), wanted):
                raise RunnerError(
                    f"{manifest_path}: config.{key}={config.get(key)!r}, expected {wanted!r}"
                )
        if manifest.get("artifact_files") != {
            "train": train_path.name,
            "val": val_path.name,
        }:
            raise RunnerError(f"{manifest_path}: artifact_files differs from fixed bundle")
        for path in (train_path, val_path, manifest_path):
            records[rel(path)] = file_record(path)

    actual = {path.resolve() for path in FEATURE_ROOT.iterdir()}
    wanted = {path.resolve() for path in expected}
    if actual != wanted:
        missing = sorted(str(path) for path in wanted - actual)
        extra = sorted(str(path) for path in actual - wanted)
        raise RunnerError(f"feature namespace is not exact; missing={missing}, extra={extra}")
    if len(records) != 15:
        raise RunnerError(f"feature snapshot has {len(records)} files, expected 15")
    return dict(sorted(records.items()))


def source_snapshot() -> dict[str, dict[str, Any]]:
    return {
        source.as_posix(): file_record(REPO_ROOT / source)
        for source in SOURCE_FILES
    }


def memory_contract() -> dict[str, Any]:
    """Validate and report the prospective parameter/state comparison before launch."""
    from lewm.models.memory import (
        HierarchicalActionConditionedMemory,
        HierarchicalActionConditionedSSMMemory,
        SSMMemory,
    )

    dimension, action_dim = 128, 6
    v5_modes = (
        "dynamic", "static", "noaction", "fixedbeta", "single", "ssmcontrol"
    )
    instances = [
        HierarchicalActionConditionedSSMMemory(dimension, action_dim, mode=mode)
        for mode in v5_modes
    ]
    signatures = [
        [[name, list(parameter.shape)] for name, parameter in model.named_parameters()]
        for model in instances
    ]
    counts = [model.parameter_count() for model in instances]
    if len(set(counts)) != 1 or counts[0] != 34_820:
        raise RunnerError(f"V5 modes are not exactly parameter matched: {counts}")
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise RunnerError("V5 mode parameter names/shapes differ")

    v4 = HierarchicalActionConditionedMemory(dimension, action_dim)
    v4_two = HierarchicalActionConditionedMemory(
        dimension, action_dim, taus=(2.0, 8.0))
    ssm = SSMMemory(dimension)
    ssm_count = sum(parameter.numel() for parameter in ssm.parameters())
    if v4.parameter_count() != 34_566 or v4_two.parameter_count() != 34_564:
        raise RunnerError("V4 parameter contract changed")
    if ssm_count != 33_024:
        raise RunnerError(f"SSM parameter contract changed: {ssm_count}")
    return {
        "embed_dim": dimension,
        "action_dim": action_dim,
        "memory_parameters": {
            "ssm": ssm_count,
            "hacsmv4": v4.parameter_count(),
            "hacsmv4_two_noaux": v4_two.parameter_count(),
            "hacssmv5_all_modes": counts[0],
        },
        "streaming_recurrent_floats": {
            "ssm": dimension,
            "hacsmv4": 3 * dimension,
            "hacsmv4_two_noaux": 2 * dimension,
            "hacssmv5_all_modes": 2 * dimension,
        },
        "v5_parameter_signature": signatures[0],
    }


def design_aux_contract(design: str) -> tuple[float, str, bool]:
    if design in V5_DESIGNS:
        return 0.05, "v5_frontload", design not in NO_AUX_DESIGNS
    if design.startswith("hacsmv4"):
        return 0.1, "fixed", design == "hacsmv4"
    if design in {"none", "ssm"}:
        return 0.0, "fixed", False
    raise RunnerError(f"no auxiliary contract for design {design!r}")


def scheduled_weight(base: float, schedule: str, epoch: int) -> float:
    if schedule == "fixed":
        return base
    if schedule != "v5_frontload" or epoch < 1:
        raise RunnerError(f"invalid hierarchy schedule/epoch: {schedule!r}/{epoch}")
    if epoch <= 20:
        return base
    if epoch <= 120:
        progress = (epoch - 20) / 100.0
        return base * 0.5 * (1.0 + math.cos(math.pi * progress))
    return 0.0


def build_protocol(commit: str, clean: bool) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "study": "HACSSM-v5 fixed-DINO hierarchical memory study",
        "producer_git_commit": commit,
        "producer_git_clean": clean,
        "common_protocol": COMMON,
        "memory_contract": memory_contract(),
        "design_protocol": {
            design: {
                "hier_loss_weight": design_aux_contract(design)[0],
                "hier_loss_schedule": design_aux_contract(design)[1],
                "auxiliary_gradients_active": design_aux_contract(design)[2],
            }
            for design in DESIGNS
        },
        "output_root": rel(OUTPUT_ROOT),
        "log_root": rel(LOG_ROOT),
        "feature_root": rel(FEATURE_ROOT),
        "feature_artifacts": feature_snapshot(),
        "source_artifacts": source_snapshot(),
        "environments": [
            {"occluded": occ, "clean_target": clean_env}
            for occ, clean_env in ENVIRONMENTS
        ],
        "stages": {
            "pilot": {
                "designs": list(DESIGNS),
                "seeds": list(PILOT_SEEDS),
                "runs": len(PILOT_JOBS),
            },
            "post_pilot_full_completion": {
                "designs": list(DESIGNS),
                "seeds": list(COMPLETION_SEEDS),
                "runs": len(COMPLETION_JOBS),
                "completed_total_runs": len(ALL_JOBS),
                "runs_regardless_of_pilot_screen": True,
            },
        },
        "analysis_gate": {
            "command": "scripts/analyze_hacssm_v5.py --phase pilot",
            "decision_file": rel(DECISION_PATH),
            "required_field": {"pilot_screen_passed": "boolean"},
            "fail_closed_result": "NO_GO",
            "if_false": "retain prospective NO_GO; still complete 120 requested cells descriptively",
            "if_true": "retain pilot pass; complete 120 requested cells and final analysis",
        },
        "expected_runs": {
            "pilot": [job.run_name for job in PILOT_JOBS],
            "completion": [job.run_name for job in COMPLETION_JOBS],
        },
    }


def establish_protocol(protocol: dict[str, Any], dry_run: bool) -> None:
    if PROTOCOL_PATH.exists():
        if not stable_equal(read_json(PROTOCOL_PATH), protocol):
            raise RunnerError(
                f"{PROTOCOL_PATH} differs from current Git/source/feature/protocol snapshot"
            )
        return
    if dry_run:
        return
    prior_entries = [
        path for path in OUTPUT_ROOT.rglob("*")
        if path.resolve() != LOCK_PATH.resolve()
    ] if OUTPUT_ROOT.exists() else []
    if LOG_ROOT.exists():
        prior_entries.extend(LOG_ROOT.rglob("*"))
    if prior_entries:
        raise RunnerError(
            f"namespace is nonempty without {PROTOCOL_PATH}: {prior_entries[:8]}"
        )
    atomic_write_json(PROTOCOL_PATH, protocol)


def expected_args(job: Job) -> dict[str, Any]:
    train_path, val_path, manifest_path = feature_paths(job.clean_env)
    base, schedule, _active = design_aux_contract(job.design)
    return {
        "env_id": job.occ_env,
        "memory_mode": job.design,
        "smt_router": COMMON["smt_router"],
        "seed": job.seed,
        "output_dir": rel(OUTPUT_ROOT),
        "num_episodes": COMMON["train_episodes"],
        "val_episodes": COMMON["val_episodes"],
        "data_dir": rel(DATA_ROOT),
        "prototype_seed": COMMON["prototype_seed"],
        "target_env_id": job.clean_env,
        "mask_occluded_target_loss": True,
        "first_post_loss_weight": COMMON["first_post_loss_weight"],
        "encoder_checkpoint": None,
        "encoder_stats": None,
        "freeze_encoder": False,
        "encoder_type": "precomputed",
        "train_feature_cache": rel(train_path),
        "val_feature_cache": rel(val_path),
        "feature_manifest": rel(manifest_path),
        "length": COMMON["length"],
        "img_size": 64,
        "epochs": COMMON["epochs"],
        "batch_size": COMMON["batch_size"],
        "lr": COMMON["learning_rate"],
        "weight_decay": COMMON["weight_decay"],
        "num_workers": COMMON["train_dataloader_workers"],
        "no_amp": False,
        "patch_size": 8,
        "embed_dim": COMMON["feature_dim"],
        "encoder_layers": 6,
        "encoder_heads": 4,
        "predictor_layers": 4,
        "predictor_heads": 8,
        "predictor_norm": COMMON["predictor_norm"],
        "history_len": COMMON["history_len"],
        "dropout": 0.1,
        "sigreg_lambda": 0.1,
        "sigreg_projections": 512,
        "hier_loss_weight": base,
        "hier_loss_schedule": schedule,
        "tau_fast": 3.0,
        "tau_slow": 25.0,
        "fixed_alpha": True,
        "wandb": False,
        "wandb_project": "lewm-memory-popgym",
        "extra_tag": "",
        "device": "cuda",
        "feature_manifest_sha256": sha256_file(manifest_path),
    }


def expected_metric_metadata(job: Job) -> dict[str, Any]:
    _train_path, _val_path, manifest_path = feature_paths(job.clean_env)
    base, schedule, active = design_aux_contract(job.design)
    final_weight = scheduled_weight(base, schedule, COMMON["epochs"])
    return {
        "env": job.occ_env,
        "design": job.design,
        "n_actions": 6,
        "prototype_seed": 0,
        "dataset_schema_version": 3,
        "feature_schema_version": 1,
        "feature_manifest": rel(manifest_path),
        "feature_manifest_sha256": sha256_file(manifest_path),
        "target_env": job.clean_env,
        "masked_clean_blackout_loss": True,
        "first_post_loss_weight": COMMON["first_post_loss_weight"],
        "hier_loss_weight": base,
        "hier_loss_schedule": schedule,
        "hier_loss_weight_final": final_weight,
        "hier_loss_weight_effective": final_weight if active else 0.0,
        "val_pred_loss_target_kind": "observed_pre_post_only",
        "deep_blackout_target_kind": "evaluation_only_hidden_clean",
        "primary_common_target_metric": "clean_mse_first_post",
        "encoder_frozen": False,
        "encoder_type": "precomputed",
        "predictor_norm": COMMON["predictor_norm"],
        "external_features_fixed": True,
        "encoder_checkpoint": None,
        "encoder_stats": None,
        "encoder_stats_sha256": None,
    }


def validate_history(history: Any, job: Job) -> None:
    if not isinstance(history, list) or len(history) != COMMON["epochs"]:
        length = len(history) if isinstance(history, list) else None
        raise RunnerError(
            f"{job.run_name}: history length {length}, expected {COMMON['epochs']}"
        )
    base, schedule, active = design_aux_contract(job.design)
    for epoch, record in enumerate(history, 1):
        if not isinstance(record, dict) or record.get("epoch") != epoch:
            raise RunnerError(f"{job.run_name}: malformed history epoch {epoch}")
        if set(record) != {"epoch", "train", "val"}:
            raise RunnerError(f"{job.run_name}: unexpected history fields at epoch {epoch}")
        for split in ("train", "val"):
            values = record.get(split)
            if not isinstance(values, dict):
                raise RunnerError(f"{job.run_name}: missing {split} history at epoch {epoch}")
            for key in ("loss", "pred_loss", "sigreg_loss"):
                value = values.get(key)
                if type(value) not in (int, float) or not math.isfinite(float(value)):
                    raise RunnerError(
                        f"{job.run_name}: invalid {split}.{key} at epoch {epoch}: {value!r}"
                    )
            assert_finite_tree(values, f"{job.run_name}.history[{epoch}].{split}")
            if job.design in HIER_DESIGNS:
                required = ["hier_loss", "hier_loss_fast", "hier_loss_medium", "hier_loss_weight"]
                if job.design.startswith("hacsmv4") and job.design != "hacsmv4_two_noaux":
                    required.append("hier_loss_slow")
                for key in required:
                    value = values.get(key)
                    if type(value) not in (int, float) or not math.isfinite(float(value)):
                        raise RunnerError(
                            f"{job.run_name}: invalid {split}.{key} at epoch {epoch}: {value!r}"
                        )
                wanted = scheduled_weight(base, schedule, epoch) if active else 0.0
                observed = float(values["hier_loss_weight"])
                if not math.isclose(observed, wanted, rel_tol=1e-6, abs_tol=1e-8):
                    raise RunnerError(
                        f"{job.run_name}: {split} hierarchy weight {observed} at epoch "
                        f"{epoch}, expected {wanted}"
                    )


def validate_model_state(state: Any, job: Job) -> None:
    import torch

    if not isinstance(state, dict) or not state:
        raise RunnerError(f"{job.run_name}: empty/non-dictionary model_state_dict")
    for name, tensor in state.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise RunnerError(f"{job.run_name}: malformed model state entry {name!r}")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise RunnerError(f"{job.run_name}: non-finite model tensor {name}")


def validate_job(job: Job, *, allow_missing: bool) -> bool:
    model_exists = job.model_path.is_file() and job.model_path.stat().st_size > 0
    metrics_exists = job.metrics_path.is_file() and job.metrics_path.stat().st_size > 0
    if model_exists != metrics_exists:
        raise RunnerError(f"partial run artifacts: {job.run_dir}")
    if not model_exists:
        if job.model_path.exists() or job.metrics_path.exists() or job.run_dir.exists():
            raise RunnerError(f"empty or incomplete run directory: {job.run_dir}")
        if allow_missing:
            return False
        raise RunnerError(f"missing required run: {job.run_dir}")

    metrics = read_json(job.metrics_path)
    if not isinstance(metrics, dict):
        raise RunnerError(f"{job.metrics_path}: expected a JSON object")
    assert_finite_tree(metrics, f"{job.run_name}.metrics")

    import torch

    try:
        checkpoint = torch.load(job.model_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RunnerError(f"cannot load checkpoint {job.model_path}: {exc}") from exc
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
        "model_state_dict", "args", "final_metrics", "history"
    }:
        keys = sorted(checkpoint) if isinstance(checkpoint, dict) else type(checkpoint).__name__
        raise RunnerError(f"{job.model_path}: unexpected checkpoint structure {keys}")
    validate_model_state(checkpoint["model_state_dict"], job)
    history = checkpoint["history"]
    validate_history(history, job)
    if not stable_equal(metrics, checkpoint["final_metrics"]):
        raise RunnerError(f"{job.run_name}: metrics.json != checkpoint final_metrics")

    wanted_args = expected_args(job)
    actual_args = checkpoint["args"]
    if not stable_equal(actual_args, wanted_args):
        if isinstance(actual_args, dict):
            differing = sorted(
                key for key in set(actual_args) | set(wanted_args)
                if key not in actual_args
                or key not in wanted_args
                or not stable_equal(actual_args[key], wanted_args[key])
            )
        else:
            differing = ["<args is not a dictionary>"]
        raise RunnerError(f"{job.run_name}: checkpoint args differ at {differing[:12]}")

    for key, wanted in expected_metric_metadata(job).items():
        if key not in metrics or not stable_equal(metrics[key], wanted):
            raise RunnerError(
                f"{job.run_name}: metric {key}={metrics.get(key)!r}, expected {wanted!r}"
            )
    required_finite = (
        "val_pred_loss",
        "infl_fast",
        "infl_slow",
        "clean_mse_deep_blackout",
        "clean_mse_deep_blackout_ablated",
        "clean_mse_first_post",
        "clean_mse_first_post_ablated",
        "constant_mse_first_post",
        "persistence_mse_first_post",
        "last_visible_mse_first_post",
        "clean_input_mse_first_post",
    )
    for key in required_finite:
        value = metrics.get(key)
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            raise RunnerError(f"{job.run_name}: invalid metric {key}={value!r}")
    parameters = metrics.get("trainable_parameters")
    if type(parameters) is not int or parameters <= 0:
        raise RunnerError(f"{job.run_name}: invalid trainable_parameters={parameters!r}")
    last_val = history[-1]["val"]
    if not stable_equal(metrics["val_pred_loss"], last_val.get("pred_loss")):
        raise RunnerError(f"{job.run_name}: val_pred_loss differs from final history")
    if job.design in HIER_DESIGNS:
        for key in ("val_hier_loss", "val_hier_loss_fast", "val_hier_loss_medium"):
            value = metrics.get(key)
            if type(value) not in (int, float) or not math.isfinite(float(value)):
                raise RunnerError(f"{job.run_name}: missing/invalid hierarchy metric {key}")
        if not stable_equal(metrics["val_hier_loss"], last_val.get("hier_loss")):
            raise RunnerError(f"{job.run_name}: val_hier_loss differs from final history")
    return True


def validate_artifact_space(jobs: Sequence[Job]) -> set[str]:
    manifest_exists = MANIFEST_PATH.is_file()
    sidecar_exists = MANIFEST_SHA_PATH.is_file()
    if manifest_exists != sidecar_exists:
        raise RunnerError("final manifest and SHA-256 sidecar must exist as a pair")
    if manifest_exists:
        manifest = read_json(MANIFEST_PATH)
        if not isinstance(manifest, dict):
            raise RunnerError(f"{MANIFEST_PATH}: expected a JSON object")
        wanted_sidecar = f"{sha256_file(MANIFEST_PATH)}  {MANIFEST_PATH.name}\n"
        try:
            observed_sidecar = MANIFEST_SHA_PATH.read_text()
        except (OSError, UnicodeError) as exc:
            raise RunnerError(f"cannot read {MANIFEST_SHA_PATH}: {exc}") from exc
        if observed_sidecar != wanted_sidecar:
            raise RunnerError(f"manifest checksum sidecar mismatch: {MANIFEST_SHA_PATH}")

    expected_names = {job.run_name for job in ALL_JOBS}
    if OUTPUT_ROOT.exists():
        unexpected_top_dirs = {
            path.name for path in OUTPUT_ROOT.iterdir()
            if path.is_dir() and path.name not in expected_names
        }
        if unexpected_top_dirs:
            raise RunnerError(f"unexpected output directories: {sorted(unexpected_top_dirs)[:8]}")
        unexpected_top_files = {
            path.name for path in OUTPUT_ROOT.iterdir()
            if path.is_file() and path.name not in TOP_LEVEL_OUTPUT_FILES
        }
        if unexpected_top_files:
            raise RunnerError(f"unexpected output files: {sorted(unexpected_top_files)[:8]}")

    expected_models = {job.model_path.resolve() for job in ALL_JOBS}
    expected_metrics = {job.metrics_path.resolve() for job in ALL_JOBS}
    actual_models = (
        {path.resolve() for path in OUTPUT_ROOT.rglob("model.pt")}
        if OUTPUT_ROOT.exists() else set()
    )
    actual_metrics = (
        {path.resolve() for path in OUTPUT_ROOT.rglob("metrics.json")}
        if OUTPUT_ROOT.exists() else set()
    )
    if actual_models - expected_models or actual_metrics - expected_metrics:
        raise RunnerError(
            "unexpected checkpoint artifacts: "
            f"models={sorted(map(str, actual_models - expected_models))[:4]}, "
            f"metrics={sorted(map(str, actual_metrics - expected_metrics))[:4]}"
        )

    expected_logs = {job.log_path.resolve() for job in ALL_JOBS}
    allowed_analysis_logs = {
        (LOG_ROOT / "analyze_pilot.log").resolve(),
        (LOG_ROOT / "analyze_final.log").resolve(),
    }
    if LOG_ROOT.exists():
        unexpected_log_entries = [path for path in LOG_ROOT.iterdir() if not path.is_file()]
        actual_logs = {path.resolve() for path in LOG_ROOT.iterdir() if path.is_file()}
        unexpected_logs = actual_logs - expected_logs - allowed_analysis_logs
        if unexpected_log_entries or unexpected_logs:
            raise RunnerError(
                f"unexpected log namespace: entries={unexpected_log_entries[:4]}, "
                f"files={sorted(map(str, unexpected_logs))[:8]}"
            )
        empty_logs = [path for path in LOG_ROOT.iterdir() if path.is_file() and path.stat().st_size <= 0]
        if empty_logs:
            raise RunnerError(f"empty log files: {empty_logs[:8]}")

    completed: set[str] = set()
    for job in jobs:
        complete = validate_job(job, allow_missing=True)
        if job.log_path.exists() and (
            not job.log_path.is_file() or job.log_path.stat().st_size <= 0
        ):
            raise RunnerError(f"empty/non-file training log: {job.log_path}")
        if not complete and job.log_path.exists():
            raise RunnerError(f"training log exists without complete run: {job.log_path}")
        if complete:
            completed.add(job.run_name)
    return completed


def train_command(python: str, job: Job) -> list[str]:
    train_path, val_path, manifest_path = feature_paths(job.clean_env)
    base, schedule, _active = design_aux_contract(job.design)
    return [
        python,
        rel(TRAIN_SCRIPT),
        "--env-id", job.occ_env,
        "--target-env-id", job.clean_env,
        "--mask-occluded-target-loss",
        "--memory-mode", job.design,
        "--smt-router", COMMON["smt_router"],
        "--seed", str(job.seed),
        "--fixed-alpha",
        "--encoder-type", "precomputed",
        "--train-feature-cache", rel(train_path),
        "--val-feature-cache", rel(val_path),
        "--feature-manifest", rel(manifest_path),
        "--prototype-seed", "0",
        "--data-dir", rel(DATA_ROOT),
        "--output-dir", rel(OUTPUT_ROOT),
        "--num-episodes", "600",
        "--val-episodes", "150",
        "--length", "32",
        "--img-size", "64",
        "--epochs", "200",
        "--batch-size", "64",
        "--lr", "3e-4",
        "--weight-decay", "1e-5",
        "--num-workers", "2",
        "--patch-size", "8",
        "--embed-dim", "128",
        "--encoder-layers", "6",
        "--encoder-heads", "4",
        "--predictor-layers", "4",
        "--predictor-heads", "8",
        "--predictor-norm", COMMON["predictor_norm"],
        "--history-len", "3",
        "--dropout", "0.1",
        "--sigreg-lambda", "0.1",
        "--sigreg-projections", "512",
        "--hier-loss-weight", str(base),
        "--hier-loss-schedule", schedule,
        "--tau-fast", "3.0",
        "--tau-slow", "25.0",
        "--first-post-loss-weight", "0.5",
        "--device", "cuda",
        "--no-wandb",
    ]


def timestamp() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def status(message: str) -> None:
    with _PRINT_LOCK:
        print(f"{timestamp()} {message}", flush=True)


def run_logged_process(
    command: Sequence[str], log_path: Path, env: dict[str, str],
    stop: threading.Event,
) -> int | None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with _PROCESS_LOCK:
        if stop.is_set():
            return None
        if log_path.exists():
            raise RunnerError(f"refusing to overwrite existing log: {log_path}")
        log = log_path.open("xb")
        try:
            process = subprocess.Popen(
                list(command), cwd=REPO_ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT,
            )
            _ACTIVE_PROCESSES.add(process)
        except BaseException:
            log.close()
            raise
    try:
        return process.wait()
    finally:
        with _PROCESS_LOCK:
            _ACTIVE_PROCESSES.discard(process)
        log.close()


def terminate_active_processes() -> None:
    with _PROCESS_LOCK:
        active = list(_ACTIVE_PROCESSES)
    for process in active:
        if process.poll() is None:
            process.terminate()
    for process in active:
        if process.poll() is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


def run_stage(
    python: str, jobs: Sequence[Job], gpu_ids: Sequence[str], workers: int
) -> None:
    for job in jobs:
        complete = validate_job(job, allow_missing=True)
        if not complete and job.log_path.exists():
            raise RunnerError(f"stale log for missing run {job.run_name}: {job.log_path}")
    shards = [tuple(jobs[slot::workers]) for slot in range(workers)]
    stop = threading.Event()

    def worker(slot: int) -> None:
        gpu = gpu_ids[slot % len(gpu_ids)]
        for job in shards[slot]:
            if stop.is_set():
                return
            if validate_job(job, allow_missing=True):
                status(f"[worker {slot} gpu {gpu}] skip validated {job.run_name}")
                continue
            status(f"[worker {slot} gpu {gpu}] >>> {job.run_name}")
            child_env = os.environ.copy()
            child_env.update({"CUDA_VISIBLE_DEVICES": gpu, "MUJOCO_GL": "egl"})
            return_code = run_logged_process(
                train_command(python, job), job.log_path, child_env, stop
            )
            if return_code is None:
                return
            if return_code != 0:
                stop.set()
                raise RunnerError(
                    f"training failed with status {return_code}: {job.run_name}; "
                    f"see {job.log_path}"
                )
            validate_job(job, allow_missing=False)
            status(f"[worker {slot} gpu {gpu}] <<< {job.run_name}")

    errors: list[BaseException] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker, slot) for slot in range(workers)]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except BaseException as exc:
                stop.set()
                terminate_active_processes()
                errors.append(exc)
    if errors:
        details = "; ".join(str(error) for error in errors[:4])
        raise RunnerError(f"stage failed in {len(errors)} worker(s): {details}") from errors[0]


def run_analyzer(python: str, phase: str) -> None:
    final = LOG_ROOT / f"analyze_{phase}.log"
    temporary = LOG_ROOT / f".analyze_{phase}.{os.getpid()}.tmp"
    if temporary.exists():
        raise RunnerError(f"stale analysis temporary log: {temporary}")
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    with temporary.open("xb") as log:
        result = subprocess.run(
            [python, rel(ANALYZE_SCRIPT), "--root", rel(OUTPUT_ROOT), "--phase", phase],
            cwd=REPO_ROOT, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, check=False,
        )
        log.flush()
        os.fsync(log.fileno())
    os.replace(temporary, final)
    if result.returncode != 0:
        raise RunnerError(
            f"{phase} analyzer failed with status {result.returncode}; see {final}"
        )


def read_pilot_decision() -> tuple[bool, dict[str, Any]]:
    decision = read_json(DECISION_PATH)
    if (not isinstance(decision, dict)
            or type(decision.get("pilot_screen_passed")) is not bool):
        raise RunnerError(
            f"{DECISION_PATH} requires a top-level boolean 'pilot_screen_passed'"
        )
    expected_label = "PILOT_PASS" if decision["pilot_screen_passed"] else "NO_GO"
    if decision.get("decision") != expected_label:
        raise RunnerError(
            f"{DECISION_PATH}: decision label conflicts with "
            f"pilot_screen_passed={decision['pilot_screen_passed']}"
        )
    assert_finite_tree(decision, "pilot_decision")
    return decision["pilot_screen_passed"], decision


def check_command_interfaces(python: str) -> None:
    for script, required in (
        (
            TRAIN_SCRIPT,
            (
                "--predictor-norm", "--hier-loss-weight", "--hier-loss-schedule",
                "v5_frontload", *DESIGNS,
            ),
        ),
        (ANALYZE_SCRIPT, ("--phase", "pilot", "final")),
    ):
        if not script.is_file():
            raise RunnerError(f"required script is missing: {script}")
        result = subprocess.run(
            [python, str(script), "--help"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            raise RunnerError(
                f"{script} --help failed ({result.returncode}): {result.stderr[-1200:]}"
            )
        help_text = result.stdout + result.stderr
        absent = [token for token in required if token not in help_text]
        if absent:
            raise RunnerError(f"{script} --help is missing required tokens: {absent}")


def check_python(python: str) -> None:
    result = subprocess.run(
        [python, "-c", "import torch; print(torch.__version__)"], cwd=REPO_ROOT,
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RunnerError(f"Python/torch preflight failed: {result.stderr.strip()}")


def check_gpus(python: str, gpu_ids: Sequence[str]) -> None:
    for gpu in dict.fromkeys(gpu_ids):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        result = subprocess.run(
            [
                python, "-c",
                "import torch; assert torch.cuda.is_available(); "
                "assert torch.cuda.device_count() == 1; print(torch.cuda.get_device_name(0))",
            ],
            cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            raise RunnerError(
                f"GPU preflight failed for CUDA_VISIBLE_DEVICES={gpu!r}: "
                f"{result.stderr.strip()}"
            )
        status(f"GPU {gpu}: {result.stdout.strip()}")


def verify_provenance_unchanged(protocol: dict[str, Any]) -> None:
    if not stable_equal(read_json(PROTOCOL_PATH), protocol):
        raise RunnerError("protocol.json changed after publication")
    if not stable_equal(source_snapshot(), protocol["source_artifacts"]):
        raise RunnerError("producer/analyzer sources changed during the study")
    if not stable_equal(feature_snapshot(), protocol["feature_artifacts"]):
        raise RunnerError("fixed feature artifacts changed during the study")
    commit, porcelain = git_provenance()
    if commit != protocol["producer_git_commit"] or porcelain:
        raise RunnerError("Git commit or clean-worktree state changed during the study")


def reject_temporary_artifacts() -> None:
    offenders = []
    for root in (OUTPUT_ROOT, LOG_ROOT):
        if root.exists():
            offenders.extend(
                path for path in root.rglob("*")
                if path.is_file() and (
                    path.name.endswith(".tmp") or path.name.startswith(".tmp")
                    or (path.name.startswith(".") and ".tmp" in path.name)
                )
            )
    if offenders:
        raise RunnerError(f"temporary/partial files remain: {offenders[:8]}")


def output_file_snapshot() -> dict[str, dict[str, Any]]:
    excluded = {LOCK_PATH.resolve(), MANIFEST_PATH.resolve(), MANIFEST_SHA_PATH.resolve()}
    return {
        rel(path): file_record(path)
        for path in sorted(OUTPUT_ROOT.rglob("*"))
        if path.is_file() and path.resolve() not in excluded
    }


def log_file_snapshot() -> dict[str, dict[str, Any]]:
    if not LOG_ROOT.exists():
        return {}
    return {
        rel(path): file_record(path)
        for path in sorted(LOG_ROOT.rglob("*")) if path.is_file()
    }


def write_final_manifest(
    protocol: dict[str, Any], decision: dict[str, Any], pilot_screen_passed: bool,
    gpu_ids: Sequence[str], workers: int,
) -> None:
    reject_temporary_artifacts()
    for job in ALL_JOBS:
        validate_job(job, allow_missing=False)
    manifest = {
        "schema_version": 1,
        "study": protocol["study"],
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "producer_git_commit": protocol["producer_git_commit"],
        "producer_git_clean": True,
        "all_requested_runs_completed": True,
        "pilot_screen_passed": pilot_screen_passed,
        "completed_runs": len(ALL_JOBS),
        "expected_runs": len(ALL_JOBS),
        "pilot_decision": decision,
        "execution": {"gpu_ids": list(gpu_ids), "workers": workers},
        "protocol": {rel(PROTOCOL_PATH): file_record(PROTOCOL_PATH)},
        "feature_artifacts": protocol["feature_artifacts"],
        "source_artifacts": protocol["source_artifacts"],
        "output_artifacts": output_file_snapshot(),
        "log_artifacts": log_file_snapshot(),
    }
    atomic_write_json(MANIFEST_PATH, manifest)
    manifest_sha = sha256_file(MANIFEST_PATH)
    atomic_write_bytes(
        MANIFEST_SHA_PATH, f"{manifest_sha}  {MANIFEST_PATH.name}\n".encode()
    )
    if sha256_file(MANIFEST_PATH) != manifest_sha:
        raise RunnerError("manifest changed immediately after atomic publication")


def parse_gpu_ids(raw: str) -> tuple[str, ...]:
    values = tuple(token.strip() for token in raw.split(",") if token.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one GPU id is required")
    for value in values:
        if any(char.isspace() for char in value) or "," in value or "=" in value:
            raise argparse.ArgumentTypeError(f"invalid CUDA device token: {value!r}")
    return values


def acquire_lock() -> Any:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stream = LOCK_PATH.open("a+")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        stream.close()
        raise RunnerError(f"another HACSSM-v5 runner holds {LOCK_PATH}") from exc
    return stream


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fixed, staged 300-cell HACSSM-v5 experiment."
    )
    parser.add_argument(
        "--python", default=str(REPO_ROOT / ".venv" / "bin" / "python"),
        help="Python executable used for training and analysis",
    )
    parser.add_argument(
        "--gpus", type=parse_gpu_ids, default=parse_gpu_ids("0,1,2,3"),
        help="comma-separated physical GPU ids (default: 0,1,2,3)",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="fixed orchestration shards, assigned round-robin to --gpus (default: 8)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="read-only interface/hash/artifact audit; launch no training or analysis",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers < 1:
        raise RunnerError("--workers must be positive")

    commit, porcelain = git_provenance()
    clean = not porcelain
    if not args.dry_run and not clean:
        preview = " | ".join(porcelain.splitlines()[:8])
        raise RunnerError(
            "actual launch requires a clean committed worktree before creating the study "
            f"namespace; dirty entries: {preview}"
        )
    if args.dry_run and not clean:
        status("DRY RUN NOTE: actual launch would fail until the current worktree is committed")

    check_python(args.python)
    check_command_interfaces(args.python)
    protocol = build_protocol(commit, clean)

    lock_stream = None
    if not args.dry_run:
        lock_stream = acquire_lock()
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        establish_protocol(protocol, args.dry_run)
        reject_temporary_artifacts()
        completed = validate_artifact_space(ALL_JOBS)
        status(
            f"preflight validated {len(completed)}/300 runs; "
            f"pilot={sum(job.run_name in completed for job in PILOT_JOBS)}/180"
        )
        if args.dry_run:
            digest = hashlib.sha256(
                json.dumps(protocol, sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
            status(
                "DRY RUN: no files written and no experiments launched; "
                f"protocol content digest={digest}"
            )
            return 0

        check_gpus(args.python, args.gpus)
        verify_provenance_unchanged(protocol)
        run_stage(args.python, PILOT_JOBS, args.gpus, args.workers)
        for job in PILOT_JOBS:
            validate_job(job, allow_missing=False)

        verify_provenance_unchanged(protocol)
        status("running pilot analyzer")
        run_analyzer(args.python, "pilot")
        pilot_screen_passed, decision = read_pilot_decision()
        status(
            f"pilot decision: {decision['decision']} "
            f"(screen_passed={pilot_screen_passed})"
        )

        status(
            "completing requested seeds 3-4; this cannot change the prospective pilot decision"
        )
        run_stage(args.python, COMPLETION_JOBS, args.gpus, args.workers)
        for job in ALL_JOBS:
            validate_job(job, allow_missing=False)
        verify_provenance_unchanged(protocol)
        status("running final five-seed analyzer")
        run_analyzer(args.python, "final")
        final_decision = read_json(FINAL_DECISION_PATH)
        allowed_final = {
            "OVERALL_BEST_IN_LOCKED_GRID", "PROMISING_NOT_OVERALL_BEST", "NO_GO",
            "PILOT_NO_GO_FINAL_DESCRIPTIVE",
        }
        if (not isinstance(final_decision, dict)
                or final_decision.get("decision") not in allowed_final
                or final_decision.get("completed_runs") != 300
                or final_decision.get("pilot_screen_passed") is not pilot_screen_passed):
            raise RunnerError(f"invalid final analyzer decision: {FINAL_DECISION_PATH}")

        verify_provenance_unchanged(protocol)
        validate_artifact_space(ALL_JOBS)
        write_final_manifest(
            protocol, decision, pilot_screen_passed, args.gpus, args.workers
        )
        status("HACSSM-v5 study complete: 300/300 validated")
        return 0
    finally:
        if lock_stream is not None:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
            lock_stream.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        terminate_active_processes()
        print("interrupted; active child processes terminated", file=sys.stderr)
        raise SystemExit(130)
    except RunnerError as exc:
        terminate_active_processes()
        print(f"HACSSM-v5 runner error: {exc}", file=sys.stderr)
        raise SystemExit(2)
