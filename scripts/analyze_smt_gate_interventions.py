"""No-retraining causal interventions on saved SMT-v2 read/write gates.

This analysis asks whether SMT-v2's sigmoid gates carry useful input- or
time-specific information beyond their average amplitude.  It evaluates every
checkpoint on the canonical matched-usage protocol from ``analyze_runs.py`` and
changes only the gate tensors:

* ``*_mean`` uses a per-bank (read) or per-channel (write) mean estimated on a
  disjoint calibration set.
* ``*_causal_time_resample`` replaces every gate after t=0 with a deterministic
  random donor from strictly earlier in the same episode.  It never moves a
  future/post-reveal gate backward in time.
* ``*_episode_shuffle`` deranges episodes while preserving each donor gate
  trajectory's time order.

The evaluation data, train/test probe split, latent targets, and shuffle maps are
identical across conditions for a checkpoint.  No parameter is changed and no
gradient is computed.  Canonical execution is intentionally fail-closed: exactly
four environments x three seeds must be present, every checkpoint must be sigmoid
SMT, all metrics must be finite, and the untouched condition must reproduce
``outputs/smt_v2/master_metrics.csv``.

Example:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
        scripts/analyze_smt_gate_interventions.py
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
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.data import generate_eval_batch  # noqa: E402
from lewm.eval.memory_probe import _fit_probe  # noqa: E402
from lewm.models.memory import TwoTimescaleMemory  # noqa: E402
from scripts.analyze_runs import build_model  # noqa: E402


ENVS = ("tmaze", "distractor", "recall", "occlusion")
SEEDS = (0, 1, 2)
CONDITIONS = (
    "original",
    "read_mean",
    "write_mean",
    "both_mean",
    "read_causal_time_resample",
    "read_episode_shuffle",
    "write_causal_time_resample",
    "write_episode_shuffle",
)
ENV_OVERRIDE_KEYS = ("reveal", "cue_len", "n_distract", "seq_len")

# These constants exactly reproduce analyze_runs.py's matched usage evaluation.
EVAL_N = 256
EVAL_SEED = 4242
PROBE_SPLIT_SEED = 0
PROBE_TRAIN_RATIO = 0.7

# Calibration is generated from a disjoint deterministic stream.
CALIB_N = 256
CALIB_SEED = 4343
INTERVENTION_SEED = 1729

USAGE_TOL = 1e-12
MSE_TOL = 2e-5
ORIGINAL_PARITY_TOL = 2e-6


PER_RUN_FIELDS = (
    "run",
    "checkpoint_sha256",
    "env",
    "seed",
    "condition",
    "router_mode",
    "n_banks",
    "taus_json",
    "eval_n",
    "eval_seed",
    "calib_n",
    "calib_seed",
    "probe_split_seed",
    "intervention_seed",
    "reveal",
    "history_len",
    "chance",
    "usage_matched",
    "val_mse",
    "delta_usage_vs_original",
    "delta_val_mse_vs_original",
    "read_gate_mean",
    "read_gate_mass",
    "write_gate_mean",
    "read_gate_mean_abs_change",
    "write_gate_mean_abs_change",
    "calib_read_mean_by_bank_json",
    "calib_write_mean",
    "calib_write_channel_std",
    "original_parity_max_abs",
    "fresh_baseline_usage",
    "fresh_baseline_val_mse",
    "canonical_usage",
    "canonical_val_mse",
)

GROUP_FIELDS = (
    "env",
    "condition",
    "n_pairs",
    "seeds_json",
    "usage_mean",
    "usage_std",
    "val_mse_mean",
    "val_mse_std",
    "paired_usage_delta_mean",
    "paired_usage_delta_std",
    "paired_val_mse_delta_mean",
    "paired_val_mse_delta_std",
    "usage_degraded_pairs",
    "mse_worsened_pairs",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministic no-retraining interventions on SMT-v2 gates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "outputs" / "smt_v2")
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "outputs" / "smt_gate_interventions"
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--encoder-batch-size",
        type=int,
        default=64,
        help="batch size used only to encode the disjoint calibration set",
    )
    args = parser.parse_args(argv)
    args.root = args.root.resolve()
    args.out = args.out.resolve()
    if args.encoder_batch_size <= 0:
        parser.error("--encoder-batch-size must be positive")
    return args


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is unavailable")
    return device


def set_canonical_numerics(seed: int) -> None:
    """Seed stochastic APIs while retaining analyze_runs.py's numerical defaults.

    In particular, cuDNN TF32 must remain enabled.  Turning it off moved one of
    77 held-out T-Maze probe decisions for seed 0, so stricter arithmetic would no
    longer be the canonical saved-checkpoint evaluation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def as_args_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise TypeError(f"checkpoint args must be a mapping/Namespace, got {type(value).__name__}")


def env_kwargs(args: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: args[key] for key in ENV_OVERRIDE_KEYS if args.get(key) is not None}


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def load_master(root: Path) -> Dict[str, Dict[str, str]]:
    path = root / "master_metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(f"canonical metric file is missing: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"run", "env", "design", "seed", "usage_matched", "val_mse"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} lacks columns: {sorted(missing)}")
        rows: Dict[str, Dict[str, str]] = {}
        for row in reader:
            run = row["run"]
            if not run or run in rows:
                raise ValueError(f"empty or duplicate run in {path}: {run!r}")
            rows[run] = dict(row)
    return rows


def discover_checkpoints(root: Path) -> list[tuple[Path, Dict[str, Any]]]:
    paths = sorted(root.glob("*/model.pt"))
    if not paths:
        raise FileNotFoundError(f"no */model.pt checkpoints below {root}")
    found: Dict[tuple[str, int], tuple[Path, Dict[str, Any]]] = {}
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if "args" not in checkpoint or "model_state_dict" not in checkpoint:
            raise ValueError(f"malformed checkpoint: {path}")
        args = as_args_dict(checkpoint["args"])
        env = str(args.get("env"))
        seed = int(args.get("seed", -1))
        key = (env, seed)
        if key in found:
            raise ValueError(f"duplicate environment/seed {key}: {found[key][0]} and {path}")
        found[key] = (path, args)
        del checkpoint

    expected = {(env, seed) for env in ENVS for seed in SEEDS}
    actual = set(found)
    if actual != expected:
        raise ValueError(
            "checkpoint matrix is not exactly 4 environments x 3 seeds; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return [found[(env, seed)] for env in ENVS for seed in SEEDS]


def validate_checkpoint_identity(
    path: Path, args: Mapping[str, Any], master: Mapping[str, Mapping[str, str]]
) -> tuple[float, float]:
    run = path.parent.name
    if args.get("memory_mode") != "smt":
        raise ValueError(f"{run}: memory_mode={args.get('memory_mode')!r}, expected 'smt'")
    if args.get("smt_router") != "sigmoid":
        raise ValueError(f"{run}: smt_router={args.get('smt_router')!r}, expected 'sigmoid'")
    row = master.get(run)
    if row is None:
        raise ValueError(f"{run}: missing from master_metrics.csv")
    expected_text = {"env": str(args["env"]), "design": "smt", "seed": str(int(args["seed"]))}
    for key, expected in expected_text.items():
        if str(row.get(key)) != expected:
            raise ValueError(
                f"{run}: canonical {key}={row.get(key)!r}, checkpoint expects {expected!r}"
            )
    canonical_usage = float(row["usage_matched"])
    canonical_mse = float(row["val_mse"])
    if not (math.isfinite(canonical_usage) and math.isfinite(canonical_mse)):
        raise ValueError(f"{run}: non-finite canonical metrics")
    return canonical_usage, canonical_mse


def make_eval_batch(args: Mapping[str, Any], n: int, seed: int) -> Dict[str, object]:
    return generate_eval_batch(
        str(args["env"]),
        n,
        img_size=int(args["img_size"]),
        length=int(args["length"]),
        seed=seed,
        **env_kwargs(args),
    )


@torch.no_grad()
def encode_in_batches(
    model: torch.nn.Module, obs: torch.Tensor, device: torch.device, batch_size: int
) -> torch.Tensor:
    chunks = []
    for start in range(0, obs.shape[0], batch_size):
        chunks.append(model.encode(obs[start : start + batch_size].to(device)))
    result = torch.cat(chunks, dim=0)
    if not torch.isfinite(result).all():
        raise FloatingPointError("non-finite calibration latents")
    return result


def cycle_derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform random cyclic derangement (no item maps to itself)."""
    if n < 2:
        raise ValueError("a derangement requires at least two items")
    order = rng.permutation(n)
    partner = np.empty(n, dtype=np.int64)
    partner[order] = np.roll(order, -1)
    if np.any(partner == np.arange(n)) or sorted(partner.tolist()) != list(range(n)):
        raise AssertionError("invalid derangement")
    return partner


def causal_time_resample_indices(
    batch: int, length: int, rng: np.random.Generator
) -> np.ndarray:
    """Past-only time donors: donor[0]=0 and donor[t] is sampled from [0,t)."""
    if length < 2:
        raise ValueError("causal time resampling requires at least two steps")
    donors = np.zeros((batch, length), dtype=np.int64)
    for episode in range(batch):
        for time in range(1, length):
            donors[episode, time] = int(rng.integers(0, time))
    target_time = np.arange(length, dtype=np.int64)[None, :]
    if np.any(donors[:, 1:] >= target_time[:, 1:]):
        raise AssertionError("causal time donor is not strictly in the past")
    return donors


def intervention_indices(batch: int, length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(INTERVENTION_SEED)
    episode = cycle_derangement(batch, rng)
    time = causal_time_resample_indices(batch, length, rng)
    # Recreate from the seed and demand byte identity: guards accidental stateful generation.
    check_rng = np.random.default_rng(INTERVENTION_SEED)
    check_episode = cycle_derangement(batch, check_rng)
    check_time = causal_time_resample_indices(batch, length, check_rng)
    if not (np.array_equal(episode, check_episode) and np.array_equal(time, check_time)):
        raise AssertionError("intervention maps are not reproducible")
    return torch.from_numpy(episode).to(device), torch.from_numpy(time).to(device)


def causal_time_resample(gates: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    expanded = index.unsqueeze(-1).expand(-1, -1, gates.shape[-1])
    return torch.gather(gates, dim=1, index=expanded)


def ensure_gate(name: str, gate: torch.Tensor, shape: tuple[int, ...]) -> None:
    if tuple(gate.shape) != shape:
        raise ValueError(f"{name} shape is {tuple(gate.shape)}, expected {shape}")
    if not torch.isfinite(gate).all():
        raise FloatingPointError(f"{name} contains non-finite values")
    lo, hi = float(gate.min()), float(gate.max())
    if lo < 0.0 or hi > 1.0:
        raise ValueError(f"{name} leaves sigmoid range: [{lo}, {hi}]")


def banks_from_write(model: torch.nn.Module, z: torch.Tensor, write: torch.Tensor) -> torch.Tensor:
    smt = model.mem_smt
    written = write * z
    return torch.stack(
        [TwoTimescaleMemory._scan(written, smt.alphas[k]) for k in range(smt.K)], dim=2
    )


def fuse_from_gates(
    model: torch.nn.Module, z: torch.Tensor, banks: torch.Tensor, read: torch.Tensor
) -> torch.Tensor:
    mixed = (read.unsqueeze(-1) * banks).sum(dim=2)
    return model.mem_smt.fuse(z, mixed)


@torch.no_grad()
def condition_metrics(
    model: torch.nn.Module,
    target_z: torch.Tensor,
    z_tilde: torch.Tensor,
    actions: torch.Tensor,
    cue: np.ndarray,
    n_classes: int,
    reveal: int,
    train_index: np.ndarray,
    test_index: np.ndarray,
    device: torch.device,
) -> tuple[float, float]:
    """Exact val_mse and matched prediction-to-prediction usage from analyze_runs.py."""
    h = int(model.history_len)
    batch, length, dim = target_z.shape
    windows = length - h
    if windows <= 0 or reveal < h or reveal >= length:
        raise ValueError(f"invalid length/history/reveal: L={length}, h={h}, reveal={reveal}")

    zt_win = (
        z_tilde.unfold(1, h, 1)[:, :windows]
        .permute(0, 1, 3, 2)
        .reshape(batch * windows, h, dim)
    )
    act = actions.to(device)
    act_win = (
        act.unfold(1, h, 1)[:, :windows]
        .permute(0, 1, 3, 2)
        .reshape(batch * windows, h, model.action_dim)
    )
    targets = target_z[:, h:length].reshape(batch * windows, dim)
    predicted = model.predictor(zt_win, act_win)[:, -1, :]
    val_mse = float(((predicted - targets) ** 2).mean())

    t = min(reveal, length - 1)
    pred_reveal = (
        model.predictor(z_tilde[:, t - h : t], act[:, t - h : t])[:, -1, :]
        .float()
        .cpu()
        .numpy()
    )
    usage = _fit_probe(
        pred_reveal[train_index],
        cue[train_index],
        pred_reveal[test_index],
        cue[test_index],
        n_classes,
    )
    if not (math.isfinite(val_mse) and math.isfinite(usage)):
        raise FloatingPointError(f"non-finite metrics: usage={usage}, val_mse={val_mse}")
    return float(usage), val_mse


@torch.no_grad()
def analyze_checkpoint(
    path: Path,
    saved_args: Mapping[str, Any],
    canonical_usage: float,
    canonical_mse: float,
    device: torch.device,
    encoder_batch_size: int,
) -> list[Dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    args = as_args_dict(checkpoint["args"])
    if args != dict(saved_args):
        raise ValueError(f"{path}: checkpoint args changed between discovery and evaluation")
    model = build_model(args).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    del checkpoint

    smt = model.mem_smt
    if smt.router_mode != "sigmoid":
        raise ValueError(f"{path.parent.name}: rebuilt router is {smt.router_mode!r}")

    calibration = make_eval_batch(args, CALIB_N, CALIB_SEED)
    evaluation = make_eval_batch(args, EVAL_N, EVAL_SEED)
    if CALIB_SEED == EVAL_SEED:
        raise AssertionError("calibration and evaluation streams must be disjoint")

    z_calib = encode_in_batches(model, calibration["obs"], device, encoder_batch_size)
    calib_read = smt.route_weights(z_calib).mean(dim=(0, 1))
    calib_write = torch.sigmoid(smt.in_gate(z_calib)).mean(dim=(0, 1))
    ensure_gate("calib_read", calib_read, (smt.K,))
    ensure_gate("calib_write", calib_write, (model.embed_dim,))
    del z_calib, calibration

    # Reproduce analyze_runs.py exactly: extract_timewise first obtains target z in
    # chunks of 64, then encode_with_memory processes the full evaluation batch to
    # produce z_tilde.  Batched and full-batch cuDNN arithmetic can differ slightly.
    target_z = encode_in_batches(model, evaluation["obs"], device, 64)
    obs = evaluation["obs"].to(device)
    actions = evaluation["actions"]
    z, _, _, reference_zt = model.encode_with_memory(obs)
    if not torch.isfinite(z).all():
        raise FloatingPointError("non-finite evaluation latents")
    batch, length, dim = z.shape
    read_original = smt.route_weights(z)
    write_original = torch.sigmoid(smt.in_gate(z))
    ensure_gate("read_original", read_original, (batch, length, smt.K))
    ensure_gate("write_original", write_original, (batch, length, dim))

    episode_index, time_index = intervention_indices(batch, length, device)
    read_mean = calib_read.view(1, 1, smt.K).expand(batch, length, smt.K)
    write_mean = calib_write.view(1, 1, dim).expand(batch, length, dim)
    read_time = causal_time_resample(read_original, time_index)
    write_time = causal_time_resample(write_original, time_index)
    read_episode = read_original.index_select(0, episode_index)
    write_episode = write_original.index_select(0, episode_index)

    read_by_condition = {
        "original": read_original,
        "read_mean": read_mean,
        "write_mean": read_original,
        "both_mean": read_mean,
        "read_causal_time_resample": read_time,
        "read_episode_shuffle": read_episode,
        "write_causal_time_resample": read_original,
        "write_episode_shuffle": read_original,
    }
    write_key = {
        "original": "original",
        "read_mean": "original",
        "write_mean": "mean",
        "both_mean": "mean",
        "read_causal_time_resample": "original",
        "read_episode_shuffle": "original",
        "write_causal_time_resample": "time",
        "write_episode_shuffle": "episode",
    }
    writes = {
        "original": write_original,
        "mean": write_mean,
        "time": write_time,
        "episode": write_episode,
    }
    bank_cache = {key: banks_from_write(model, z, value) for key, value in writes.items()}

    # Verify that our explicit gate path is the checkpoint's untouched implementation.
    original_zt = fuse_from_gates(model, z, bank_cache["original"], read_original)
    parity_max = float((original_zt - reference_zt).abs().max())
    if not math.isfinite(parity_max) or parity_max > ORIGINAL_PARITY_TOL:
        raise AssertionError(
            f"explicit/original SMT path differs by {parity_max} (tol={ORIGINAL_PARITY_TOL})"
        )

    cue = evaluation["cue"].numpy()
    n_classes = int(evaluation["n_cue_classes"])
    reveal = int(evaluation["reveal"].float().mean())
    split_rng = np.random.default_rng(PROBE_SPLIT_SEED)
    permutation = split_rng.permutation(len(cue))
    n_train = int(PROBE_TRAIN_RATIO * len(cue))
    train_index, test_index = permutation[:n_train], permutation[n_train:]
    if len(np.unique(cue[train_index])) < 2:
        raise ValueError("canonical probe training split has fewer than two cue classes")

    # A fresh canonical baseline under the exact same numerical settings is the
    # authoritative reference for the explicit intervention path.
    fresh_usage, fresh_mse = condition_metrics(
        model,
        target_z,
        reference_zt,
        actions,
        cue,
        n_classes,
        reveal,
        train_index,
        test_index,
        device,
    )

    base_columns = {
        "run": path.parent.name,
        "checkpoint_sha256": sha256_file(path),
        "env": str(args["env"]),
        "seed": int(args["seed"]),
        "router_mode": smt.router_mode,
        "n_banks": smt.K,
        "taus_json": json.dumps([float(x) for x in smt.taus]),
        "eval_n": EVAL_N,
        "eval_seed": EVAL_SEED,
        "calib_n": CALIB_N,
        "calib_seed": CALIB_SEED,
        "probe_split_seed": PROBE_SPLIT_SEED,
        "intervention_seed": INTERVENTION_SEED,
        "reveal": reveal,
        "history_len": int(model.history_len),
        "chance": 1.0 / n_classes,
        "calib_read_mean_by_bank_json": json.dumps(
            [float(x) for x in calib_read.detach().cpu()]
        ),
        "calib_write_mean": float(calib_write.mean()),
        "calib_write_channel_std": float(calib_write.std(unbiased=False)),
        "original_parity_max_abs": parity_max,
        "fresh_baseline_usage": fresh_usage,
        "fresh_baseline_val_mse": fresh_mse,
        "canonical_usage": canonical_usage,
        "canonical_val_mse": canonical_mse,
    }

    raw_rows: list[Dict[str, Any]] = []
    for condition in CONDITIONS:
        read = read_by_condition[condition]
        write = writes[write_key[condition]]
        ensure_gate(f"{condition}:read", read, (batch, length, smt.K))
        ensure_gate(f"{condition}:write", write, (batch, length, dim))
        z_tilde = (
            original_zt
            if condition == "original"
            else fuse_from_gates(model, z, bank_cache[write_key[condition]], read)
        )
        usage, val_mse = condition_metrics(
            model,
            target_z,
            z_tilde,
            actions,
            cue,
            n_classes,
            reveal,
            train_index,
            test_index,
            device,
        )
        raw_rows.append(
            {
                **base_columns,
                "condition": condition,
                "usage_matched": usage,
                "val_mse": val_mse,
                "read_gate_mean": float(read.mean()),
                "read_gate_mass": float(read.sum(dim=-1).mean()),
                "write_gate_mean": float(write.mean()),
                "read_gate_mean_abs_change": float((read - read_original).abs().mean()),
                "write_gate_mean_abs_change": float((write - write_original).abs().mean()),
            }
        )

    original = next(row for row in raw_rows if row["condition"] == "original")
    if abs(float(original["usage_matched"]) - fresh_usage) > USAGE_TOL:
        raise AssertionError(
            f"{path.parent.name}: explicit usage {original['usage_matched']} != fresh canonical "
            f"usage {fresh_usage}"
        )
    if abs(float(original["val_mse"]) - fresh_mse) > MSE_TOL:
        raise AssertionError(
            f"{path.parent.name}: explicit MSE {original['val_mse']} != fresh canonical "
            f"MSE {fresh_mse}"
        )
    if abs(float(original["usage_matched"]) - canonical_usage) > USAGE_TOL:
        raise AssertionError(
            f"{path.parent.name}: original usage {original['usage_matched']} != canonical "
            f"{canonical_usage} (tol={USAGE_TOL})"
        )
    if abs(float(original["val_mse"]) - canonical_mse) > MSE_TOL:
        raise AssertionError(
            f"{path.parent.name}: original MSE {original['val_mse']} != canonical "
            f"{canonical_mse} (tol={MSE_TOL})"
        )
    for row in raw_rows:
        row["delta_usage_vs_original"] = float(row["usage_matched"]) - float(
            original["usage_matched"]
        )
        row["delta_val_mse_vs_original"] = float(row["val_mse"]) - float(
            original["val_mse"]
        )
        if set(row) != set(PER_RUN_FIELDS):
            raise AssertionError(f"internal CSV schema mismatch: {set(row) ^ set(PER_RUN_FIELDS)}")
    return raw_rows


def sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise ValueError("sample standard deviation requires at least two values")
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def group_rows(rows: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    groups: Dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["env"]), str(row["condition"]))].append(row)
    expected_keys = {(env, condition) for env in ENVS for condition in CONDITIONS}
    if set(groups) != expected_keys:
        raise ValueError(f"group matrix mismatch: {set(groups) ^ expected_keys}")

    output = []
    for env in ENVS:
        for condition in CONDITIONS:
            part = sorted(groups[(env, condition)], key=lambda row: int(row["seed"]))
            seeds = [int(row["seed"]) for row in part]
            if seeds != list(SEEDS):
                raise ValueError(f"{env}/{condition}: expected seeds {SEEDS}, found {seeds}")
            usage = [float(row["usage_matched"]) for row in part]
            mse = [float(row["val_mse"]) for row in part]
            du = [float(row["delta_usage_vs_original"]) for row in part]
            dm = [float(row["delta_val_mse_vs_original"]) for row in part]
            grouped = {
                "env": env,
                "condition": condition,
                "n_pairs": len(part),
                "seeds_json": json.dumps(seeds),
                "usage_mean": float(np.mean(usage)),
                "usage_std": sample_std(usage),
                "val_mse_mean": float(np.mean(mse)),
                "val_mse_std": sample_std(mse),
                "paired_usage_delta_mean": float(np.mean(du)),
                "paired_usage_delta_std": sample_std(du),
                "paired_val_mse_delta_mean": float(np.mean(dm)),
                "paired_val_mse_delta_std": sample_std(dm),
                "usage_degraded_pairs": sum(value < 0 for value in du),
                "mse_worsened_pairs": sum(value > 0 for value in dm),
            }
            if set(grouped) != set(GROUP_FIELDS):
                raise AssertionError("internal grouped CSV schema mismatch")
            if not all(
                math.isfinite(float(grouped[field]))
                for field in GROUP_FIELDS
                if field
                not in {"env", "condition", "seeds_json", "usage_degraded_pairs", "mse_worsened_pairs"}
            ):
                raise FloatingPointError(f"non-finite grouped metrics for {env}/{condition}")
            output.append(grouped)
    return output


def print_summary(grouped: Sequence[Mapping[str, Any]]) -> None:
    print("\nPaired changes versus original (mean ± sample SD over 3 seeds)")
    print(f"{'env':<12} {'condition':<23} {'delta usage':>19} {'delta MSE':>19}")
    for row in grouped:
        if row["condition"] == "original":
            continue
        print(
            f"{row['env']:<12} {row['condition']:<23} "
            f"{row['paired_usage_delta_mean']:>8.4f} ± {row['paired_usage_delta_std']:<7.4f} "
            f"{row['paired_val_mse_delta_mean']:>8.4f} ± {row['paired_val_mse_delta_std']:<7.4f}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    atomic_write_text(
        args.out / "status.json",
        json.dumps({"status": "running", "started_utc": started}, indent=2) + "\n",
    )
    try:
        device = resolve_device(args.device)
        set_canonical_numerics(INTERVENTION_SEED)
        master = load_master(args.root)
        checkpoints = discover_checkpoints(args.root)
        print(
            f"Analyzing {len(checkpoints)} sigmoid-SMT checkpoints on {device}; "
            f"eval={EVAL_N}@{EVAL_SEED}, calibration={CALIB_N}@{CALIB_SEED}"
        )

        rows: list[Dict[str, Any]] = []
        checkpoint_manifest = []
        for index, (path, saved_args) in enumerate(checkpoints, start=1):
            canonical_usage, canonical_mse = validate_checkpoint_identity(path, saved_args, master)
            print(f"[{index:02d}/{len(checkpoints):02d}] {path.parent.name}", flush=True)
            run_rows = analyze_checkpoint(
                path,
                saved_args,
                canonical_usage,
                canonical_mse,
                device,
                args.encoder_batch_size,
            )
            rows.extend(run_rows)
            original = next(row for row in run_rows if row["condition"] == "original")
            print(
                f"  original usage={original['usage_matched']:.4f}, "
                f"MSE={original['val_mse']:.6f}; 8/8 conditions complete",
                flush=True,
            )
            checkpoint_manifest.append(
                {
                    "run": path.parent.name,
                    "path": str(path),
                    "sha256": original["checkpoint_sha256"],
                }
            )
            del run_rows
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if len(rows) != len(ENVS) * len(SEEDS) * len(CONDITIONS):
            raise AssertionError(f"expected 96 per-run rows, found {len(rows)}")
        grouped = group_rows(rows)
        if len(grouped) != len(ENVS) * len(CONDITIONS):
            raise AssertionError(f"expected 32 grouped rows, found {len(grouped)}")

        per_run_path = args.out / "per_run_metrics.csv"
        grouped_path = args.out / "grouped_paired_metrics.csv"
        manifest_path = args.out / "manifest.json"
        atomic_write_csv(per_run_path, PER_RUN_FIELDS, rows)
        atomic_write_csv(grouped_path, GROUP_FIELDS, grouped)
        completed = datetime.now(timezone.utc).isoformat()
        manifest = {
            "status": "complete",
            "started_utc": started,
            "completed_utc": completed,
            "root": str(args.root),
            "output": str(args.out),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "numerical_backend": {
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
            },
            "reproducibility_caveat": (
                "A diagnostic run with cuDNN TF32 disabled changed T-Maze seed-0 usage "
                "from 0.9350649351 to 0.9220779221 (one of 77 held-out episodes). "
                "Canonical analyze_runs.py backend defaults are therefore preserved. "
                "The explicit gate path is additionally required to match a fresh canonical "
                "path in every checkpoint."
            ),
            "protocol": {
                "usage": (
                    "analyze_runs.py: usage_matched; eval_n=256; eval_seed=4242; "
                    "split_seed=0; train_ratio=0.7; prediction-to-prediction probe"
                ),
                "val_mse": "analyze_runs.py: all sliding history windows against encoder targets",
                "calibration": (
                    "256 disjoint generated episodes at seed 4343; read per-bank mean and "
                    "write per-channel mean over episodes and time"
                ),
                "causal_time_resample": (
                    "within each episode, t=0 is unchanged and every later gate uses a "
                    "deterministic random donor from a strictly earlier time; future and "
                    "post-reveal gates are never moved backward"
                ),
                "episode_shuffle": (
                    "single across-episode cyclic derangement preserving donor time order; "
                    "no fixed episodes"
                ),
                "std": "sample standard deviation over the 3 paired checkpoint seeds (ddof=1)",
            },
            "constants": {
                "eval_n": EVAL_N,
                "eval_seed": EVAL_SEED,
                "calib_n": CALIB_N,
                "calib_seed": CALIB_SEED,
                "probe_split_seed": PROBE_SPLIT_SEED,
                "intervention_seed": INTERVENTION_SEED,
                "conditions": list(CONDITIONS),
            },
            "checkpoints": checkpoint_manifest,
            "files": {
                "per_run": str(per_run_path),
                "grouped": str(grouped_path),
            },
        }
        atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        atomic_write_text(
            args.out / "status.json",
            json.dumps(
                {
                    "status": "complete",
                    "started_utc": started,
                    "completed_utc": completed,
                    "rows": len(rows),
                    "groups": len(grouped),
                },
                indent=2,
            )
            + "\n",
        )
        print_summary(grouped)
        print(f"\nWrote {per_run_path}")
        print(f"Wrote {grouped_path}")
        print(f"Wrote {manifest_path}")
        return 0
    except BaseException as exc:
        failed = datetime.now(timezone.utc).isoformat()
        atomic_write_text(
            args.out / "status.json",
            json.dumps(
                {
                    "status": "failed",
                    "started_utc": started,
                    "failed_utc": failed,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                indent=2,
            )
            + "\n",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
