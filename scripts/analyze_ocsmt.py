"""Analyze deterministic OC-SMT gate usage on generated evaluation episodes.

For every ``*/model.pt`` below an output root, this script rebuilds the model with
``scripts.analyze_runs.build_model``, generates the checkpoint's evaluation
environment (including any saved environment overrides), and evaluates the
hard-concrete gates on latents produced by the trained encoder.

Three CSVs are written by default:

* ``ocsmt_gate_metrics.csv``: one row per checkpoint.
* ``ocsmt_gate_metrics_grouped.csv``: population mean/std over seeds for each
  environment/configuration/L0 weight.
* ``ocsmt_gate_failures.csv``: an empty manifest on success, or one row per failure.

``master_metrics.csv`` is required by default so that its canonical matched usage
probe can be reported beside the gate statistics. Usage is deliberately not
recomputed here: keeping the value from ``analyze_runs.py`` prevents subtly different
probe splits or definitions from being mixed in the same table.

Example:
    python scripts/analyze_ocsmt.py outputs/ocsmt --eval-n 128 --device cuda
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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse the canonical checkpoint-to-model constructor.  In particular, this keeps
# compatibility with all architecture defaults used by the main analysis script.
from scripts.analyze_runs import build_model  # noqa: E402
from lewm.data import generate_eval_batch  # noqa: E402


ENV_OVERRIDE_KEYS = ("reveal", "cue_len", "n_distract", "seq_len")
MASTER_USAGE_COLUMN = "usage_matched"
MASTER_METADATA_FIELDS = ("run", "env", "design", "seed", "suffix", "exp", "length")
MASTER_USAGE_PROTOCOL = (
    "analyze_runs.py:usage_matched;eval_n=256;eval_seed=4242;"
    "split_seed=0;train_ratio=0.7;probe=prediction_to_prediction"
)

# These fields identify a run without changing the learned outcome. Everything else
# saved in checkpoint['args'] participates in the outcome-configuration fingerprint.
# In particular, optimizer/training/model settings such as gate_lr_mult, epochs, LR,
# encoder initialization, and history length must never be silently pooled.
OUTCOME_CONFIG_EXCLUDED_FIELDS = frozenset(
    {
        "seed",
        "output_dir",
        "run_suffix",
        "extra_tag",
        "wandb",
        "wandb_project",
        "wandb_entity",
    }
)

PER_RUN_FIELDS = (
    "run",
    "env",
    "design",
    "seed",
    "suffix",
    "exp",
    "config_fingerprint",
    "l0_lambda",
    "oc_num",
    "length",
    "img_size",
    "reveal_override",
    "cue_len_override",
    "n_distract_override",
    "seq_len_override",
    "eval_n",
    "eval_seed",
    "active_threshold",
    "analysis_batch_size",
    "analysis_device",
    "usage_matched",
    "usage_source",
    "usage_protocol",
    "mean_active_count",
    "active_fraction",
    "active_count_std",
    "active_count_temporal_std",
    "active_count_input_std",
    "active_count_episode_std",
    "active_count_min",
    "active_count_max",
    "mean_gate_mass",
    "gate_mass_std",
    "mean_gate_value",
    "gate_overall_std",
    "gate_temporal_std",
    "gate_input_std",
    "gate_within_episode_temporal_std",
    "gate_episode_mean_std",
    "expected_open_count",
    "expected_open_count_within_eval_std",
)

# Config fields are intentionally more specific than merely (env, lambda).  This
# prevents runs with different sequence lengths or environment difficulty overrides
# from being silently pooled into one seed aggregate.
GROUP_FIELDS = (
    "env",
    "l0_lambda",
    "oc_num",
    "length",
    "img_size",
    "reveal_override",
    "cue_len_override",
    "n_distract_override",
    "seq_len_override",
    "config_fingerprint",
)

GROUP_CONTEXT_FIELDS = (
    "eval_n",
    "eval_seed",
    "active_threshold",
    "analysis_batch_size",
    "analysis_device",
    "usage_source",
    "usage_protocol",
)

FAILURE_FIELDS = ("run", "checkpoint", "error_type", "error")

METRIC_FIELDS = (
    "usage_matched",
    "mean_active_count",
    "active_fraction",
    "active_count_std",
    "active_count_temporal_std",
    "active_count_input_std",
    "active_count_episode_std",
    "mean_gate_mass",
    "gate_mass_std",
    "mean_gate_value",
    "gate_overall_std",
    "gate_temporal_std",
    "gate_input_std",
    "gate_within_episode_temporal_std",
    "gate_episode_mean_std",
    "expected_open_count",
    "expected_open_count_within_eval_std",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze deterministic gates in an OC-SMT checkpoint directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        help="output root containing per-run directories (positional form)",
    )
    parser.add_argument(
        "--root",
        dest="root_option",
        type=Path,
        help="output root (equivalent to the positional argument)",
    )
    parser.add_argument("-n", "--eval-n", type=int, default=128, help="evaluation episodes per run")
    parser.add_argument("--seed", type=int, default=4242, help="generated evaluation-data seed")
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device, e.g. auto, cpu, cuda, or cuda:1",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="encoder batch size (does not change the generated evaluation set)",
    )
    parser.add_argument(
        "--active-threshold",
        type=float,
        default=0.0,
        help="a deterministic gate is active iff gate > this value",
    )
    parser.add_argument(
        "--per-run-csv",
        type=Path,
        default=None,
        help="per-run output path (full-run default: ROOT/ocsmt_gate_metrics.csv)",
    )
    parser.add_argument(
        "--grouped-csv",
        type=Path,
        default=None,
        help="grouped output path (full-run default: ROOT/ocsmt_gate_metrics_grouped.csv)",
    )
    parser.add_argument(
        "--failures-csv",
        type=Path,
        default=None,
        help="failure manifest path (full-run default: ROOT/ocsmt_gate_failures.csv)",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="write successful rows and return zero even if some checkpoints fail",
    )
    parser.add_argument(
        "--allow-stale-master",
        action="store_true",
        help="allow master_metrics.csv to predate a selected checkpoint (recorded as stale)",
    )
    parser.add_argument(
        "--allow-missing-usage",
        action="store_true",
        help="allow a missing/non-finite canonical usage_matched value and emit NaN",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "analyze only the first N sorted checkpoints; implicit outputs use "
            "non-canonical *.limit-N.csv names"
        ),
    )
    args = parser.parse_args(argv)

    if args.root is not None and args.root_option is not None:
        parser.error("provide ROOT either positionally or with --root, not both")
    args.root = (args.root_option or args.root or (REPO_ROOT / "outputs" / "ocsmt")).resolve()
    del args.root_option

    if args.eval_n <= 0:
        parser.error("--eval-n must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if not math.isfinite(args.active_threshold) or args.active_threshold < 0:
        parser.error("--active-threshold must be finite and non-negative")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested ({spec}) but CUDA is unavailable")
    return device


def set_reproducible(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Gate evaluation itself has no sampling, but these settings also make encoder
    # execution repeatable on the usual CUDA kernels used by this repository.
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def as_args_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise TypeError(f"checkpoint 'args' must be a mapping or Namespace, got {type(value).__name__}")


def _canonical_config_value(value: Any) -> Any:
    """Convert checkpoint arguments to a deterministic JSON representation."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite float in checkpoint configuration: {value!r}")
        return value
    if isinstance(value, np.generic):
        return _canonical_config_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"non-string checkpoint configuration key: {key!r}")
            out[key] = _canonical_config_value(item)
        return out
    if isinstance(value, (list, tuple)):
        return [_canonical_config_value(item) for item in value]
    raise TypeError(
        "unsupported checkpoint configuration value "
        f"{value!r} ({type(value).__name__})"
    )


def outcome_config_fingerprint(checkpoint_args: Mapping[str, Any]) -> str:
    """Hash every saved outcome-affecting argument except seed/run/logging identity."""
    config: Dict[str, Any] = {}
    for key, value in checkpoint_args.items():
        if not isinstance(key, str):
            raise TypeError(f"non-string checkpoint argument key: {key!r}")
        if key not in OUTCOME_CONFIG_EXCLUDED_FIELDS:
            config[key] = _canonical_config_value(value)
    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def env_kwargs(checkpoint_args: Mapping[str, Any]) -> Dict[str, Any]:
    """Return exactly the environment overrides persisted by train_memory.py."""
    return {
        key: checkpoint_args[key]
        for key in ENV_OVERRIDE_KEYS
        if checkpoint_args.get(key) is not None
    }


def eval_config(checkpoint_args: Mapping[str, Any]) -> Tuple[Any, ...]:
    overrides = env_kwargs(checkpoint_args)
    return (
        checkpoint_args["env"],
        int(checkpoint_args["img_size"]),
        int(checkpoint_args["length"]),
        *(overrides.get(key) for key in ENV_OVERRIDE_KEYS),
    )


def generate_checkpoint_eval(
    checkpoint_args: Mapping[str, Any], eval_n: int, eval_seed: int
) -> Dict[str, object]:
    return generate_eval_batch(
        checkpoint_args["env"],
        eval_n,
        img_size=int(checkpoint_args["img_size"]),
        length=int(checkpoint_args["length"]),
        seed=eval_seed,
        **env_kwargs(checkpoint_args),
    )


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_master_usage(root: Path) -> Tuple[Dict[str, Dict[str, str]], str, str]:
    """Load canonical usage plus identity metadata from master_metrics.csv."""
    path = root / "master_metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(f"required usage file does not exist: {path}")

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        required = set(MASTER_METADATA_FIELDS) | {MASTER_USAGE_COLUMN}
        missing = sorted(required - fieldnames)
        if missing:
            raise ValueError(f"{path} is missing required column(s): {', '.join(missing)}")
        usage: Dict[str, Dict[str, str]] = {}
        for row in reader:
            run = row.get("run", "")
            if not run:
                continue
            if run in usage:
                raise ValueError(f"duplicate run {run!r} in {path}")
            usage[run] = dict(row)
    return usage, f"master_metrics.csv:{MASTER_USAGE_COLUMN}", MASTER_USAGE_PROTOCOL


def master_is_stale(master_path: Path, checkpoints: Sequence[Path]) -> Tuple[bool, List[Path]]:
    """Return selected checkpoints newer than master_metrics.csv."""
    master_mtime = master_path.stat().st_mtime_ns
    newer = [path for path in checkpoints if path.stat().st_mtime_ns > master_mtime]
    return bool(newer), newer


def _csv_int(row: Mapping[str, Any], field: str, run: str) -> int:
    value = _float_or_nan(row.get(field))
    if not math.isfinite(value) or not float(value).is_integer():
        raise ValueError(f"master row {run!r} has invalid integer {field}={row.get(field)!r}")
    return int(value)


def validated_master_usage(
    checkpoint_path: Path,
    checkpoint_args: Mapping[str, Any],
    usage_rows: Mapping[str, Mapping[str, Any]],
    allow_missing: bool,
) -> float:
    """Validate the master row identity before returning finite canonical usage."""
    run = checkpoint_path.parent.name
    row = usage_rows.get(run)
    if row is None:
        if allow_missing:
            return float("nan")
        raise ValueError(f"run {run!r} is missing from master_metrics.csv")

    expected_text = {
        "run": run,
        "env": str(checkpoint_args["env"]),
        "design": str(checkpoint_args["memory_mode"]),
        "suffix": str(checkpoint_args.get("run_suffix", "")),
        "exp": str(checkpoint_args.get("extra_tag", "")),
    }
    for field, expected in expected_text.items():
        actual = str(row.get(field, ""))
        if actual != expected:
            raise ValueError(
                f"master metadata mismatch for {run!r}: {field}={actual!r}, "
                f"checkpoint expects {expected!r}"
            )

    expected_int = {
        "seed": int(checkpoint_args.get("seed", 0)),
        "length": int(checkpoint_args["length"]),
    }
    for field, expected in expected_int.items():
        actual = _csv_int(row, field, run)
        if actual != expected:
            raise ValueError(
                f"master metadata mismatch for {run!r}: {field}={actual}, "
                f"checkpoint expects {expected}"
            )

    value = _float_or_nan(row.get(MASTER_USAGE_COLUMN))
    if not math.isfinite(value):
        if allow_missing:
            return float("nan")
        raise ValueError(
            f"master row {run!r} has non-finite {MASTER_USAGE_COLUMN}="
            f"{row.get(MASTER_USAGE_COLUMN)!r}"
        )
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"master row {run!r} has out-of-range {MASTER_USAGE_COLUMN}={value}; "
            "expected an accuracy in [0, 1]"
        )
    return value


@torch.inference_mode()
def collect_gate_tensors(
    model: torch.nn.Module,
    observations: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic gates (B,L,M) and expected-open counts (B,L) on CPU."""
    memory = getattr(model, "mem_ocsmt", None)
    if memory is None:
        raise TypeError("checkpoint did not build an OC-SMT model (missing model.mem_ocsmt)")

    gate_parts: List[torch.Tensor] = []
    expected_parts: List[torch.Tensor] = []
    for start in range(0, observations.shape[0], batch_size):
        obs = observations[start : start + batch_size].to(device, non_blocking=True)
        z = model.encode(obs)
        logits = memory.gate(z)
        # Derive the deterministic hard-concrete gate from the already-computed
        # logits instead of evaluating memory.gate(z) a second time.
        gates = (
            torch.sigmoid(logits) * (memory.ZETA - memory.GAMMA) + memory.GAMMA
        ).clamp(0, 1)
        p_open = torch.sigmoid(
            logits - memory.BETA * math.log(-memory.GAMMA / memory.ZETA)
        )
        gate_parts.append(gates.float().cpu())
        expected_parts.append(p_open.sum(dim=-1).float().cpu())
        del obs, z, logits, gates, p_open
    return torch.cat(gate_parts, dim=0), torch.cat(expected_parts, dim=0)


def gate_statistics(
    gates: torch.Tensor,
    expected_counts: torch.Tensor,
    threshold: float,
) -> Dict[str, float]:
    """Compute complementary deterministic and expected gate-size statistics.

    ``gate_temporal_std`` uses the raw OC-SMT gates: average episodes first, then
    take each bank's population std over time and average over banks.  Unlike
    ``smt_router_viz.py``, gates are deliberately not normalized across banks,
    because total gate mass/cardinality is part of the OC-SMT quantity of interest.
    ``gate_input_std`` instead takes the episode std at each (time, bank) and
    averages those values. The two distinguish a shared time schedule from
    input/episode-dependent routing. Additional within-episode and episode-mean
    statistics make that decomposition explicit.
    """
    if gates.ndim != 3:
        raise ValueError(f"expected gates with shape (B,L,M), got {tuple(gates.shape)}")
    if expected_counts.shape != gates.shape[:2]:
        raise ValueError(
            "expected-open counts must have shape (B,L), got "
            f"{tuple(expected_counts.shape)} for gates {tuple(gates.shape)}"
        )
    if gates.numel() == 0:
        raise ValueError("cannot summarize an empty gate tensor")
    if not bool(torch.isfinite(gates).all()):
        bad = int((~torch.isfinite(gates)).sum())
        raise ValueError(f"gate tensor contains {bad} non-finite value(s)")
    if not bool(torch.isfinite(expected_counts).all()):
        bad = int((~torch.isfinite(expected_counts)).sum())
        raise ValueError(f"expected-open counts contain {bad} non-finite value(s)")

    # CPU float64 reductions make CSV values stable and avoid precision loss when
    # measuring the very small temporal/input variation seen in these routers.
    g = gates.to(dtype=torch.float64)
    expected = expected_counts.to(dtype=torch.float64)
    active = (g > threshold).sum(dim=-1).to(dtype=torch.float64)  # (B,L)
    mass = g.sum(dim=-1)  # deterministic soft cardinality, (B,L)
    M = g.shape[-1]

    temporal_by_bank = g.mean(dim=0).std(dim=0, unbiased=False)  # (M,)
    input_at_time_bank = g.std(dim=0, unbiased=False)  # (L,M)
    within_episode_time = g.std(dim=1, unbiased=False)  # (B,M)
    episode_mean_by_bank = g.mean(dim=1).std(dim=0, unbiased=False)  # (M,)

    active_temporal = active.mean(dim=0).std(unbiased=False)
    active_input = active.std(dim=0, unbiased=False).mean()
    active_episode = active.mean(dim=1).std(unbiased=False)

    return {
        "mean_active_count": float(active.mean()),
        "active_fraction": float(active.mean() / M),
        "active_count_std": float(active.std(unbiased=False)),
        "active_count_temporal_std": float(active_temporal),
        "active_count_input_std": float(active_input),
        "active_count_episode_std": float(active_episode),
        "active_count_min": float(active.min()),
        "active_count_max": float(active.max()),
        "mean_gate_mass": float(mass.mean()),
        "gate_mass_std": float(mass.std(unbiased=False)),
        "mean_gate_value": float(g.mean()),
        "gate_overall_std": float(g.std(unbiased=False)),
        "gate_temporal_std": float(temporal_by_bank.mean()),
        "gate_input_std": float(input_at_time_bank.mean()),
        "gate_within_episode_temporal_std": float(within_episode_time.mean()),
        "gate_episode_mean_std": float(episode_mean_by_bank.mean()),
        "expected_open_count": float(expected.mean()),
        "expected_open_count_within_eval_std": float(expected.std(unbiased=False)),
    }


def analyze_checkpoint(
    checkpoint_path: Path,
    checkpoint: Mapping[str, Any],
    checkpoint_args: Mapping[str, Any],
    observations: torch.Tensor,
    usage_value: float,
    usage_source: str,
    usage_protocol: str,
    device: torch.device,
    eval_n: int,
    eval_seed: int,
    batch_size: int,
    threshold: float,
) -> Dict[str, Any]:
    args = dict(checkpoint_args)
    if args.get("memory_mode") != "ocsmt":
        raise TypeError(f"memory_mode={args.get('memory_mode')!r}, expected 'ocsmt'")

    model = build_model(args)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    gates, expected = collect_gate_tensors(model, observations, device, batch_size)
    stats = gate_statistics(gates, expected, threshold)

    run = checkpoint_path.parent.name
    overrides = env_kwargs(args)
    row: Dict[str, Any] = {
        "run": run,
        "env": args["env"],
        "design": args["memory_mode"],
        "seed": int(args.get("seed", 0)),
        "suffix": args.get("run_suffix", ""),
        "exp": args.get("extra_tag", ""),
        "config_fingerprint": outcome_config_fingerprint(args),
        "l0_lambda": float(args.get("l0_lambda", 0.0)),
        "oc_num": int(args.get("oc_num", gates.shape[-1])),
        "length": int(args["length"]),
        "img_size": int(args["img_size"]),
        "reveal_override": overrides.get("reveal", ""),
        "cue_len_override": overrides.get("cue_len", ""),
        "n_distract_override": overrides.get("n_distract", ""),
        "seq_len_override": overrides.get("seq_len", ""),
        "eval_n": eval_n,
        "eval_seed": eval_seed,
        "active_threshold": threshold,
        "analysis_batch_size": batch_size,
        "analysis_device": str(device),
        "usage_matched": usage_value,
        "usage_source": usage_source,
        "usage_protocol": usage_protocol,
        **stats,
    }
    del model, gates, expected
    return row


def _finite(values: Iterable[Any]) -> np.ndarray:
    array = np.asarray([_float_or_nan(value) for value in values], dtype=np.float64)
    return array[np.isfinite(array)]


def group_rows(rows: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    groups: MutableMapping[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in GROUP_FIELDS)].append(row)

    fields = list(GROUP_FIELDS) + list(GROUP_CONTEXT_FIELDS) + [
        "n_runs",
        "n_seeds",
        "seed_list",
    ]
    for metric in METRIC_FIELDS:
        fields.extend((f"{metric}_n", f"{metric}_mean", f"{metric}_std"))

    grouped: List[Dict[str, Any]] = []
    for key in sorted(groups, key=lambda item: tuple(str(value) for value in item)):
        members = groups[key]
        out: Dict[str, Any] = dict(zip(GROUP_FIELDS, key))
        runs_by_seed: MutableMapping[int, List[str]] = defaultdict(list)
        for member in members:
            runs_by_seed[int(member["seed"])].append(str(member["run"]))
        duplicates = {
            seed: runs for seed, runs in runs_by_seed.items() if len(runs) > 1
        }
        if duplicates:
            detail = "; ".join(
                f"seed {seed}: {', '.join(runs)}" for seed, runs in sorted(duplicates.items())
            )
            raise ValueError(
                f"duplicate seed(s) in config group {out['config_fingerprint']}: {detail}"
            )

        for field in GROUP_CONTEXT_FIELDS:
            first = members[0][field]
            if any(member[field] != first for member in members[1:]):
                raise ValueError(
                    f"mixed {field!r} in config group {out['config_fingerprint']}"
                )
            out[field] = first

        seeds = sorted(runs_by_seed)
        out["n_runs"] = len(members)
        out["n_seeds"] = len(seeds)
        out["seed_list"] = json.dumps(seeds, separators=(",", ":"))
        for metric in METRIC_FIELDS:
            values = _finite(row.get(metric) for row in members)
            out[f"{metric}_n"] = int(values.size)
            out[f"{metric}_mean"] = float(values.mean()) if values.size else float("nan")
            # Population std matches the repository's existing aggregation scripts.
            out[f"{metric}_std"] = float(values.std(ddof=0)) if values.size else float("nan")
        grouped.append(out)
    return grouped, fields


def write_csv_atomic(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with tmp.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _mean_std_cell(row: Mapping[str, Any], metric: str, digits: int = 3) -> str:
    n = int(row[f"{metric}_n"])
    if n == 0:
        return "-"
    mean = float(row[f"{metric}_mean"])
    std = float(row[f"{metric}_std"])
    return f"{mean:.{digits}f}+/-{std:.{digits}f}"


def print_table(grouped: Sequence[Mapping[str, Any]]) -> None:
    print("\nOC-SMT deterministic gate analysis (population mean+/-std over unique seeds)")
    print(
        f"{'env':<12} {'lambda0':>8} {'M':>4} {'n':>3} "
        f"{'usage':>17} {'active':>18} {'mass':>18} {'E[open]':>18} "
        f"{'gate_tstd':>18} {'gate_istd':>18} {'active_sd':>18}"
    )
    for row in grouped:
        print(
            f"{str(row['env']):<12} {float(row['l0_lambda']):>8.4g} "
            f"{int(row['oc_num']):>4} {int(row['n_seeds']):>3} "
            f"{_mean_std_cell(row, 'usage_matched'):>17} "
            f"{_mean_std_cell(row, 'mean_active_count'):>18} "
            f"{_mean_std_cell(row, 'mean_gate_mass'):>18} "
            f"{_mean_std_cell(row, 'expected_open_count'):>18} "
            f"{_mean_std_cell(row, 'gate_temporal_std', 5):>18} "
            f"{_mean_std_cell(row, 'gate_input_std', 5):>18} "
            f"{_mean_std_cell(row, 'active_count_std'):>18}"
        )


def failure_row(
    path: Path | None, exc: BaseException, run: str | None = None
) -> Dict[str, str]:
    return {
        "run": run if run is not None else (path.parent.name if path is not None else ""),
        "checkpoint": str(path) if path is not None else "",
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.root.is_dir():
        print(f"error: output root does not exist: {args.root}", file=sys.stderr)
        return 2

    limit_suffix = f".limit-{args.limit}" if args.limit is not None else ""
    per_run_default = args.root / f"ocsmt_gate_metrics{limit_suffix}.csv"
    grouped_default = args.root / f"ocsmt_gate_metrics_grouped{limit_suffix}.csv"
    failures_default = args.root / f"ocsmt_gate_failures{limit_suffix}.csv"
    per_run_path = (args.per_run_csv or per_run_default).resolve()
    grouped_path = (args.grouped_csv or grouped_default).resolve()
    failures_path = (args.failures_csv or failures_default).resolve()
    output_paths = {per_run_path, grouped_path, failures_path}
    if len(output_paths) != 3:
        print("error: per-run, grouped, and failures CSV paths must be distinct", file=sys.stderr)
        return 2
    master_path = (args.root / "master_metrics.csv").resolve()
    if master_path in output_paths:
        print(
            f"error: output CSV path would overwrite input usage file: {master_path}",
            file=sys.stderr,
        )
        return 2

    try:
        device = resolve_device(args.device)
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    set_reproducible(args.seed)

    all_checkpoints = sorted(args.root.glob("*/model.pt"))
    checkpoint_inputs = {path.resolve() for path in all_checkpoints}
    collisions = output_paths & checkpoint_inputs
    if collisions:
        joined = ", ".join(str(path) for path in sorted(collisions))
        print(f"error: output CSV path would overwrite checkpoint input(s): {joined}", file=sys.stderr)
        return 2
    checkpoints = all_checkpoints
    if args.limit is not None:
        checkpoints = checkpoints[: args.limit]
    if not checkpoints:
        print(f"error: no */model.pt checkpoints found under {args.root}", file=sys.stderr)
        return 2

    failures: List[Dict[str, str]] = []
    usage_rows: Dict[str, Dict[str, str]] = {}
    usage_source = "unavailable"
    usage_protocol = "unavailable"
    try:
        if master_path.is_file():
            stale, newer = master_is_stale(master_path, checkpoints)
            if stale:
                sample = ", ".join(path.parent.name for path in newer[:3])
                suffix = "" if len(newer) <= 3 else f", ... ({len(newer)} total)"
                message = (
                    f"{master_path} predates selected checkpoint(s): {sample}{suffix}; "
                    "rerun scripts/analyze_runs.py or pass --allow-stale-master"
                )
                if not args.allow_stale_master:
                    raise RuntimeError(message)
                print(f"warning: {message}", file=sys.stderr)
            usage_rows, usage_source, usage_protocol = load_master_usage(args.root)
            if stale:
                usage_source += ";stale_master_allowed=true"
        elif args.allow_missing_usage:
            print(
                f"warning: {master_path} is missing; usage_matched will be NaN",
                file=sys.stderr,
            )
        else:
            raise FileNotFoundError(f"required usage file does not exist: {master_path}")
    except Exception as exc:
        failures.append(failure_row(master_path, exc, run="<master_metrics>"))
        write_csv_atomic(failures_path, FAILURE_FIELDS, failures)
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"wrote {failures_path} ({len(failures)} failure)", file=sys.stderr)
        return 1

    print(
        f"analyzing {len(checkpoints)} checkpoint(s) from {args.root} "
        f"on {device} with eval_n={args.eval_n}, seed={args.seed}"
    )
    rows: List[Dict[str, Any]] = []
    cached_config: Tuple[Any, ...] | None = None
    cached_batch: Dict[str, object] | None = None

    for checkpoint_path in checkpoints:
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            checkpoint_args = as_args_dict(checkpoint["args"])
            usage_value = validated_master_usage(
                checkpoint_path,
                checkpoint_args,
                usage_rows,
                allow_missing=args.allow_missing_usage,
            )
            config = eval_config(checkpoint_args)
            if config != cached_config:
                cached_batch = generate_checkpoint_eval(checkpoint_args, args.eval_n, args.seed)
                cached_config = config
            assert cached_batch is not None
            row = analyze_checkpoint(
                checkpoint_path=checkpoint_path,
                checkpoint=checkpoint,
                checkpoint_args=checkpoint_args,
                observations=cached_batch["obs"],
                usage_value=usage_value,
                usage_source=usage_source,
                usage_protocol=usage_protocol,
                device=device,
                eval_n=args.eval_n,
                eval_seed=args.seed,
                batch_size=args.batch_size,
                threshold=args.active_threshold,
            )
            rows.append(row)
            print(
                f"  {row['run']}: active={row['mean_active_count']:.3f}/{row['oc_num']} "
                f"mass={row['mean_gate_mass']:.3f} E[open]={row['expected_open_count']:.3f}"
            )
            del checkpoint
        except Exception as exc:  # continue so one corrupt run does not discard all results
            failure = failure_row(checkpoint_path, exc)
            failures.append(failure)
            print(
                f"  SKIP {failure['run']}: {failure['error_type']}: {failure['error']}",
                file=sys.stderr,
            )

    write_csv_atomic(failures_path, FAILURE_FIELDS, failures)

    if failures and not args.allow_partial:
        print(
            f"error: {len(failures)} checkpoint(s) failed; result CSVs were not overwritten "
            f"(pass --allow-partial to opt in)",
            file=sys.stderr,
        )
        print(f"wrote {failures_path} ({len(failures)} failures)", file=sys.stderr)
        return 1

    if not rows:
        print("error: no OC-SMT checkpoints were analyzed successfully", file=sys.stderr)
        return 1

    rows.sort(
        key=lambda row: (
            str(row["env"]),
            float(row["l0_lambda"]),
            int(row["seed"]),
            str(row["run"]),
        )
    )
    try:
        grouped, grouped_fields = group_rows(rows)
    except Exception as exc:
        failures.append(failure_row(None, exc, run="<grouping>"))
        write_csv_atomic(failures_path, FAILURE_FIELDS, failures)
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"wrote {failures_path} ({len(failures)} failures)", file=sys.stderr)
        return 1

    write_csv_atomic(per_run_path, PER_RUN_FIELDS, rows)
    write_csv_atomic(grouped_path, grouped_fields, grouped)

    print_table(grouped)
    print(f"\nwrote {per_run_path} ({len(rows)} runs)")
    print(f"wrote {grouped_path} ({len(grouped)} groups)")
    print(f"wrote {failures_path} ({len(failures)} failures)")
    if failures:
        print(
            f"warning: skipped {len(failures)} checkpoint(s) under --allow-partial",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
