#!/usr/bin/env python3
"""Probes on the V19 P2 eval exports (docs/V19_PROPOSAL.md 4.5; no GPU).

For every ``eval_export.npz`` written by scripts/train_v19_p2.py this script
computes, per (task, arm, seed):

- **xi-coordinate probe on prior_read** at the registered evaluation times.
  Categorical tasks: the REGISTERED coordinate is the mean of ``prior_read``
  over the deep-gap/post-cue window ``[cue_off + 2 .. t_dec]`` (for the shell
  game the informative phase ends at ``shuffle_off``, mirroring
  ``lewm.tasks_v19.certify._postcue_start``) concatenated with ``prior_read``
  at ``t_dec``; logistic accuracy at ``t_dec`` alone and averaged over the
  last 8 pre-decision steps are reported alongside.  T4: ridge R^2 on xi from
  ``prior_read`` at ``t = gap_off``, plus the posterior-mean and
  frozen-position reference distances (the unobserved-evolution check).
- **checkpoint integrator floor**: ridge/logistic from
  ``[enc(o_0), a_{t-3:t-1}, sum a, t/(L-1)]`` — the certify.py feature style
  with the encoder embedding replacing simulator ground truth.

Probes are trained on one half of the val episodes and evaluated on the
other, averaged over 3 probe seeds (the seed draws the split).  Results land
in ``probe_results.json`` next to each export; scripts/aggregate_v19_p2.py
builds the study summary and power analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.tasks_v19.certify import POSTCUE_OFFSET, _cat_accuracy, _ridge_r2

PROBE_SEEDS = (0, 1, 2)
PROBE_SPLIT_SALT = 1_000
LAST_K_PRE_DECISION = 8
RESULTS_NAME = "probe_results.json"
SCHEMA_VERSION = 1


# --------------------------------------------------------------------------
# Export IO
# --------------------------------------------------------------------------

def load_export(path: str | Path) -> dict[str, Any]:
    """Load an eval export: arrays plus the parsed ``meta`` dict."""
    with np.load(Path(path)) as data:
        export: dict[str, Any] = {name: data[name] for name in data.files}
    export["meta"] = json.loads(str(export.pop("meta_json")))
    return export


def export_events(export: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {name.removeprefix("event_"): value
            for name, value in export.items() if name.startswith("event_")}


# --------------------------------------------------------------------------
# Feature builders (registered coordinates)
# --------------------------------------------------------------------------

def deep_window_start(events: Mapping[str, np.ndarray]) -> np.ndarray:
    """First frame of the deep-gap/post-cue window (per episode).

    ``shuffle_off`` supersedes ``cue_off`` for the shell game, mirroring the
    P1a non-re-observability convention (certify._postcue_start).
    """
    if "shuffle_off" in events:
        return np.asarray(events["shuffle_off"], dtype=np.int64) + POSTCUE_OFFSET
    return np.asarray(events["cue_off"], dtype=np.int64) + POSTCUE_OFFSET


def registered_cat_features(prior_read: np.ndarray,
                            window_start: np.ndarray) -> np.ndarray:
    """(E, 2D): mean prior_read over [window_start .. t_dec] ++ prior_read[t_dec]."""
    episodes, length, _ = prior_read.shape
    t_dec = length - 1
    steps = np.arange(length)[None, :]
    mask = (steps >= window_start[:, None]) & (steps <= t_dec)
    if not mask.any(axis=1).all():
        raise ValueError("empty deep-gap window for at least one episode")
    weights = mask.astype(np.float64) / mask.sum(axis=1, keepdims=True)
    window_mean = np.einsum("el,eld->ed", weights, prior_read.astype(np.float64))
    return np.concatenate(
        [window_mean, prior_read[:, t_dec].astype(np.float64)], axis=1)


def integrator_floor_features(enc_o0: np.ndarray,
                              actions: np.ndarray) -> np.ndarray:
    """Checkpoint integrator features: [enc(o_0), a_{t-3:t-1}, sum a, t/(L-1)].

    The certify.integrator_features layout with the frozen-encoder embedding
    of the initial observation replacing simulator ground truth.
    """
    episodes = enc_o0.shape[0]
    length = actions.shape[1] + 1
    t_dec = length - 1
    return np.concatenate([
        enc_o0.astype(np.float64),
        actions[:, t_dec - 3:t_dec].reshape(episodes, -1),
        actions.sum(axis=1),
        np.full((episodes, 1), t_dec / (length - 1), dtype=np.float64),
    ], axis=1)


def _split(episodes: int, probe_seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Half/half episode split drawn by the probe seed."""
    order = np.random.default_rng(PROBE_SPLIT_SALT + probe_seed).permutation(
        episodes)
    half = episodes // 2
    return order[:half], order[half:]


# --------------------------------------------------------------------------
# Probes
# --------------------------------------------------------------------------

def _probe_cat(export: Mapping[str, Any], probe_seed: int) -> dict[str, float]:
    prior_read = export["prior_read"]
    xi = export["xi"]
    events = export_events(export)
    length = prior_read.shape[1]
    t_dec = length - 1
    train_idx, eval_idx = _split(prior_read.shape[0], probe_seed)

    registered = registered_cat_features(prior_read, deep_window_start(events))
    results = {
        "registered": _cat_accuracy(registered[train_idx], xi[train_idx],
                                    registered[eval_idx], xi[eval_idx]),
        "t_dec": _cat_accuracy(prior_read[train_idx, t_dec], xi[train_idx],
                               prior_read[eval_idx, t_dec], xi[eval_idx]),
    }
    last_scores = [
        _cat_accuracy(prior_read[train_idx, t], xi[train_idx],
                      prior_read[eval_idx, t], xi[eval_idx])
        for t in range(t_dec - LAST_K_PRE_DECISION, t_dec)]
    results["last8"] = float(np.mean(last_scores))

    floor = integrator_floor_features(export["enc_o0"], export["actions"])
    results["floor"] = _cat_accuracy(floor[train_idx], xi[train_idx],
                                     floor[eval_idx], xi[eval_idx])
    return results


def _probe_cont(export: Mapping[str, Any], probe_seed: int) -> dict[str, float]:
    prior_read = export["prior_read"]
    xi = export["xi"]
    events = export_events(export)
    episodes = prior_read.shape[0]
    gap_off = np.asarray(events["gap_off"], dtype=np.int64)
    features = prior_read[np.arange(episodes), gap_off]
    train_idx, eval_idx = _split(episodes, probe_seed)

    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    probe = make_pipeline(StandardScaler(), Ridge(alpha=1e-3))
    probe.fit(features[train_idx], xi[train_idx])
    prediction = probe.predict(features[eval_idx])

    posterior = export["posterior_mean"][eval_idx]
    frozen = export["frozen_pos"][eval_idx]
    dist_posterior = float(np.linalg.norm(
        prediction - posterior, axis=1).mean())
    dist_frozen = float(np.linalg.norm(prediction - frozen, axis=1).mean())
    floor = integrator_floor_features(export["enc_o0"], export["actions"])
    return {
        "registered": float(r2_score(xi[eval_idx], prediction)),
        "posterior_mean_r2": float(r2_score(xi[eval_idx], posterior)),
        "dist_to_posterior_mean": dist_posterior,
        "dist_to_frozen": dist_frozen,
        "closer_to_posterior": float(dist_posterior < dist_frozen),
        "floor": _ridge_r2(floor[train_idx], xi[train_idx],
                           floor[eval_idx], xi[eval_idx]),
    }


def run_probes(export: Mapping[str, Any],
               probe_seeds: Iterable[int] = PROBE_SEEDS) -> dict[str, Any]:
    """Full probe battery on one export, averaged over probe seeds."""
    meta = export["meta"]
    probe_seeds = tuple(probe_seeds)
    probe = _probe_cat if meta["xi_kind"] == "cat" else _probe_cont
    per_seed = [probe(export, seed) for seed in probe_seeds]
    keys = per_seed[0].keys()
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task": meta["task"],
        "arm": meta["arm"],
        "seed": meta["seed"],
        "host": meta["host"],
        "xi_kind": meta["xi_kind"],
        "n_classes": meta["n_classes"],
        "chance": (1.0 / meta["n_classes"] if meta["xi_kind"] == "cat" else 0.0),
        "probe_seeds": list(probe_seeds),
        "metric": "accuracy" if meta["xi_kind"] == "cat" else "r2",
    }
    for key in keys:
        values = [float(result[key]) for result in per_seed]
        summary[key] = {"mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "per_probe_seed": values}
    summary["memory_advantage"] = (summary["registered"]["mean"]
                                   - summary["floor"]["mean"])
    return summary


# --------------------------------------------------------------------------
# Discovery / main
# --------------------------------------------------------------------------

def discover_exports(root: str | Path) -> list[Path]:
    return sorted(Path(root).glob("*/*/s*/eval_export.npz"))


def process_run(export_path: Path, probe_seeds: Iterable[int] = PROBE_SEEDS,
                force: bool = False) -> dict[str, Any]:
    results_path = export_path.parent / RESULTS_NAME
    if results_path.exists() and not force:
        return json.loads(results_path.read_text())
    results = run_probes(load_export(export_path), probe_seeds)
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    return results


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/v19_p2")
    parser.add_argument("--probe-seeds", default="0,1,2")
    parser.add_argument("--force", action="store_true",
                        help="recompute even if probe_results.json exists")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    probe_seeds = tuple(int(seed) for seed in args.probe_seeds.split(","))
    exports = discover_exports(args.root)
    if not exports:
        raise FileNotFoundError(f"no eval_export.npz under {args.root}")
    for export_path in exports:
        results = process_run(export_path, probe_seeds, force=args.force)
        print(f"[v19-p2-eval] {results['task']}/{results['arm']}/"
              f"s{results['seed']}: xi={results['registered']['mean']:.4f} "
              f"floor={results['floor']['mean']:.4f} "
              f"(chance={results['chance']:.3f})", flush=True)


if __name__ == "__main__":
    main()
