#!/usr/bin/env python3
"""Logistic-readout confirmation for completed DINO-WM Wall carrier cells.

The exploratory Wall grid used a deterministic ridge readout to avoid the
high-dimensional LBFGS bottleneck.  This script does not retrain carriers and
does not overwrite the exploratory cells.  It reloads trained carriers, reuses
the existing Wall feature cache, fits the registered logistic readout, and
writes a separate confirmation directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.frozen_swap_carriers import make_frozen_carrier  # noqa: E402
from scripts.run_dinowm_wall_stage_h import (  # noqa: E402
    AGES,
    ARMS,
    DEFAULT_CHECKPOINT,
    DEFAULT_HYDRA,
    DEFAULT_OUTPUT as DEFAULT_SOURCE,
    DEFAULT_VENDOR,
    FrozenWallHost,
    WallFeatureBank,
    device_from,
    evaluate_cell,
    set_seed,
    sha256_file,
    stable_json,
)

DEFAULT_OUTPUT = ROOT / "outputs/dinowm_wall_audit_v1/stage_h_logistic_readout"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--hydra", type=Path, default=DEFAULT_HYDRA)
    parser.add_argument("--vendor", type=Path, default=DEFAULT_VENDOR)
    parser.add_argument("--arms", nargs="+", choices=ARMS,
                        default=["ssm", "fixed_trust"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--ages", nargs="+", type=int, default=list(AGES))
    parser.add_argument("--pca-dim", type=int, default=256)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    return parser.parse_args()


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in ("source", "output", "checkpoint", "hydra", "vendor"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    args.ages = tuple(int(age) for age in args.ages)
    if any(age not in AGES for age in args.ages):
        raise ValueError(f"--ages must be a subset of {AGES}, got {args.ages}")
    if args.pca_dim < 1:
        raise ValueError("--pca-dim must be positive")
    args.readout = f"pca{args.pca_dim}_lr"
    args.epochs = 0
    return args


def fit_pca_logistic(args: argparse.Namespace, train_x: np.ndarray,
                     train_y: np.ndarray):
    """Registered dimension-reduced logistic readout.

    The exploratory grid read from 8064-D multiscale pooled DINO features.
    Direct LBFGS over that space is too slow for confirmation.  This readout
    keeps the logistic classifier but predeclares a PCA bottleneck fit only on
    the training split.
    """

    components = min(int(args.pca_dim), int(train_x.shape[0]) - 1,
                     int(train_x.shape[1]))
    if components < 1:
        raise ValueError("not enough samples/features for PCA readout")
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=components, svd_solver="randomized", random_state=0),
        LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=int(args.max_iter),
            random_state=0,
        ),
    ).fit(train_x, train_y)


def load_carrier(args: argparse.Namespace, arm: str, seed: int,
                 device: torch.device) -> tuple[torch.nn.Module, Path]:
    carrier_path = args.source / "cells" / arm / f"s{seed}" / "carrier.pt"
    if not carrier_path.exists():
        raise FileNotFoundError(carrier_path)
    payload = torch.load(carrier_path, map_location="cpu", weights_only=False)
    carrier = make_frozen_carrier(arm, 384, 10).to(device)
    carrier.load_state_dict(payload["carrier_state_dict"])
    carrier.eval()
    return carrier, carrier_path


def write_cell(args: argparse.Namespace, arm: str, seed: int,
               metrics: dict[str, Any], arrays: dict[str, np.ndarray],
               carrier_path: Path) -> None:
    cell = args.output / "cells" / arm / f"s{seed}"
    if cell.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {cell}")
    cell.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cell / "predictions.npz", **arrays)
    manifest = {
        "schema": "dinowm_wall_stage_h_logistic_readout_cell_v1",
        "status": "complete",
        "arm": arm,
        "seed": int(seed),
        "source_carrier": {
            "path": str(carrier_path.relative_to(ROOT)),
            "sha256": sha256_file(carrier_path),
        },
        "readout": args.readout,
        "readout_contract": {
            "scaler": "StandardScaler",
            "reducer": "PCA",
            "pca_dim": int(args.pca_dim),
            "classifier": "LogisticRegression(lbfgs)",
            "max_iter": int(args.max_iter),
            "fit_on": "training split only",
        },
        "metrics": metrics,
        "artifacts": {"predictions": "predictions.npz"},
    }
    (cell / "metrics.json").write_text(stable_json(manifest))


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "schema": "dinowm_wall_stage_h_logistic_readout_summary_v1",
        "status": "complete",
        "source": str(args.source.relative_to(ROOT)),
        "readout": f"pca{args.pca_dim}_lr",
        "readout_contract": {
            "scaler": "StandardScaler",
            "reducer": "PCA",
            "pca_dim": int(args.pca_dim),
            "classifier": "LogisticRegression(lbfgs)",
            "max_iter": int(args.max_iter),
            "fit_on": "training split only",
        },
        "ages": list(args.ages),
        "arms": {},
    }
    for arm in args.arms:
        rows = []
        for seed in args.seeds:
            path = args.output / "cells" / arm / f"s{seed}" / "metrics.json"
            if path.exists():
                rows.append(json.loads(path.read_text()))
        if not rows:
            continue
        ages = {}
        for age in args.ages:
            key = str(age)
            full = [r["metrics"][key]["full"]["balanced_accuracy"]
                    for r in rows]
            reset = [r["metrics"][key]["reset_with_full_readout"]
                     ["balanced_accuracy"] for r in rows]
            no_state = [r["metrics"][key]["no_state_with_full_readout"]
                        ["balanced_accuracy"] for r in rows]
            prior = [r["metrics"][key]["prior"]["balanced_accuracy"]
                     for r in rows]
            ages[key] = {
                "full_mean": float(np.mean(full)),
                "full_values": [float(x) for x in full],
                "reset_mean": float(np.mean(reset)),
                "reset_values": [float(x) for x in reset],
                "no_state_mean": float(np.mean(no_state)),
                "no_state_values": [float(x) for x in no_state],
                "prior_mean": float(np.mean(prior)),
                "prior_values": [float(x) for x in prior],
                "passes": bool(
                    np.mean(full) >= 0.75
                    and np.mean(reset) <= 0.30
                    and np.mean(no_state) <= 0.30),
            }
        summary["arms"][arm] = {
            "seeds": [int(r["seed"]) for r in rows],
            "ages": ages,
        }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(stable_json(summary))
    return summary


def main() -> None:
    args = resolve_args(parse_args())
    args.output.mkdir(parents=True, exist_ok=True)
    if args.aggregate:
        summary = aggregate(args)
        print(stable_json({
            "status": "aggregate_complete",
            "summary": str((args.output / "summary.json").relative_to(ROOT)),
            "arms": sorted(summary["arms"]),
        }))
        return
    set_seed(0)
    evaluate_cell.__globals__["fit_readout"] = fit_pca_logistic
    evaluate_cell.__globals__["AGES"] = tuple(args.ages)
    device = device_from(args)
    host = FrozenWallHost(args, device)
    bank = WallFeatureBank(args.source)
    for arm in args.arms:
        for seed in args.seeds:
            set_seed(seed)
            args.arm = arm
            args.seed = int(seed)
            carrier, carrier_path = load_carrier(args, arm, seed, device)
            metrics, arrays = evaluate_cell(host, carrier, bank, args)
            write_cell(args, arm, seed, metrics, arrays, carrier_path)
            print(stable_json({
                "status": "cell_complete",
                "arm": arm,
                "seed": int(seed),
                "output": str((args.output / "cells" / arm
                               / f"s{seed}" / "metrics.json")
                              .relative_to(ROOT)),
            }))
    summary = aggregate(args)
    print(stable_json({
        "status": "complete",
        "summary": str((args.output / "summary.json").relative_to(ROOT)),
        "arms": sorted(summary["arms"]),
    }))


if __name__ == "__main__":
    main()
