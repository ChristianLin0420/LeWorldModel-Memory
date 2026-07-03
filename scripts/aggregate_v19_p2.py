#!/usr/bin/env python3
"""Aggregate V19 P2 probe results into the study summary + power analysis.

Reads every ``probe_results.json`` written by scripts/eval_v19_p2.py under
the P2 root and emits ``p2_summary.json`` and ``p2_summary.md`` with, per
task:

- arm x {xi-probe (registered coordinate), checkpoint integrator floor},
  mean +- std over training seeds;
- paired arm-vs-envelope differences with per-seed pairing, where the
  envelope is the per-seed better of Ac-GRU / Ac-SSM (the action-conditioned
  recurrent envelope, proposal Tier 1);
- a POWER ANALYSIS from the seed-level paired LKC-vs-envelope differences:
  bootstrap the seed-mean at n_seeds in {3, 5, 8, 10} (one-sided one-sample
  t-test at alpha = 0.05 on resamples of the centered differences shifted to
  the target effect) and report the smallest n with >= 80% power for the
  observed effect and for a registered +5% effect.  Reported per task and
  pooled over tasks (each (task, seed) pair is one sample).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_v19_p2 import RESULTS_NAME

ENVELOPE_ARMS = ("acgru", "acssm")
CANDIDATE_ARM = "lkc"
N_SEED_GRID = (3, 5, 8, 10)
N_BOOTSTRAP = 5_000
ALPHA = 0.05
POWER_TARGET = 0.80
REGISTERED_EFFECT = 0.05
POWER_RNG_SEED = 20_260_703


# --------------------------------------------------------------------------
# Collection
# --------------------------------------------------------------------------

def collect(root: str | Path) -> dict[str, dict[str, dict[int, dict[str, float]]]]:
    """{task: {arm: {seed: {'xi': ..., 'floor': ..., 'chance': ...}}}}."""
    table: dict[str, dict[str, dict[int, dict[str, float]]]] = {}
    for path in sorted(Path(root).glob(f"*/*/s*/{RESULTS_NAME}")):
        results = json.loads(path.read_text())
        cell = table.setdefault(results["task"], {}).setdefault(
            results["arm"], {})
        cell[int(results["seed"])] = {
            "xi": float(results["registered"]["mean"]),
            "floor": float(results["floor"]["mean"]),
            "chance": float(results["chance"]),
            "metric": results["metric"],
        }
    if not table:
        raise FileNotFoundError(f"no {RESULTS_NAME} files under {root}")
    return table


def _mean_std(values: list[float]) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "std": float(np.std(values)),
            "n": len(values)}


def envelope_by_seed(arms: dict[str, dict[int, dict[str, float]]]
                     ) -> dict[int, float]:
    """Per-seed better of the available Ac-GRU / Ac-SSM xi scores."""
    envelope: dict[int, float] = {}
    for arm in ENVELOPE_ARMS:
        for seed, cell in arms.get(arm, {}).items():
            envelope[seed] = max(envelope.get(seed, -np.inf), cell["xi"])
    return envelope


def paired_differences(arms: dict[str, dict[int, dict[str, float]]]
                       ) -> dict[str, dict[str, Any]]:
    """Per arm: seed-paired xi difference vs the recurrent envelope."""
    envelope = envelope_by_seed(arms)
    output: dict[str, dict[str, Any]] = {}
    for arm, cells in sorted(arms.items()):
        if arm in ENVELOPE_ARMS:
            continue
        seeds = sorted(set(cells) & set(envelope))
        if not seeds:
            continue
        diffs = [cells[seed]["xi"] - envelope[seed] for seed in seeds]
        output[arm] = {
            "seeds": seeds,
            "differences": diffs,
            **_mean_std(diffs),
            "wins": int(sum(diff > 0 for diff in diffs)),
        }
    return output


# --------------------------------------------------------------------------
# Power analysis
# --------------------------------------------------------------------------

def _power_at(diffs: np.ndarray, effect: float, n_seeds: int,
              rng: np.random.Generator, n_bootstrap: int = N_BOOTSTRAP,
              alpha: float = ALPHA) -> float:
    """P(one-sided one-sample t-test rejects mean<=0) for n_seeds resamples
    of the observed differences re-centered at ``effect``."""
    centered = diffs - diffs.mean() + effect
    samples = rng.choice(centered, size=(n_bootstrap, n_seeds), replace=True)
    means = samples.mean(axis=1)
    sds = samples.std(axis=1, ddof=1)
    critical = stats.t.ppf(1.0 - alpha, df=n_seeds - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = means / (sds / np.sqrt(n_seeds))
    # Degenerate resamples (zero spread): reject iff the mean is positive.
    t_stat = np.where(sds == 0.0, np.where(means > 0, np.inf, -np.inf), t_stat)
    return float((t_stat > critical).mean())


def power_analysis(diffs: list[float],
                   n_seed_grid: Iterable[int] = N_SEED_GRID) -> dict[str, Any]:
    """Bootstrap power over the seed grid for the observed and +5% effects."""
    diffs_array = np.asarray(diffs, dtype=np.float64)
    if diffs_array.size < 2:
        return {"status": "insufficient_seed_pairs", "n_pairs": diffs_array.size}
    rng = np.random.default_rng(POWER_RNG_SEED)
    observed = float(diffs_array.mean())
    analysis: dict[str, Any] = {
        "status": "ok",
        "n_pairs": int(diffs_array.size),
        "observed_effect": observed,
        "observed_sd": float(diffs_array.std(ddof=1)),
        "alpha": ALPHA,
        "power_target": POWER_TARGET,
        "n_bootstrap": N_BOOTSTRAP,
        "effects": {},
    }
    for label, effect in (("observed", observed),
                          ("registered_plus_5pct", REGISTERED_EFFECT)):
        powers = {int(n): _power_at(diffs_array, effect, int(n), rng)
                  for n in n_seed_grid}
        adequate = [n for n, power in powers.items() if power >= POWER_TARGET]
        analysis["effects"][label] = {
            "effect": float(effect),
            "power_by_n_seeds": powers,
            "smallest_n_with_80pct_power": min(adequate) if adequate else None,
        }
    return analysis


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

def aggregate(root: str | Path) -> dict[str, Any]:
    table = collect(root)
    summary: dict[str, Any] = {"schema_version": 1, "root": str(root),
                               "tasks": {}}
    pooled_lkc_diffs: list[float] = []
    for task, arms in sorted(table.items()):
        task_summary: dict[str, Any] = {"arms": {}, "paired_vs_envelope": {}}
        for arm, cells in sorted(arms.items()):
            seeds = sorted(cells)
            task_summary["arms"][arm] = {
                "seeds": seeds,
                "xi_probe": _mean_std([cells[seed]["xi"] for seed in seeds]),
                "integrator_floor": _mean_std(
                    [cells[seed]["floor"] for seed in seeds]),
                "chance": cells[seeds[0]]["chance"],
                "metric": cells[seeds[0]]["metric"],
            }
        paired = paired_differences(arms)
        task_summary["paired_vs_envelope"] = paired
        if CANDIDATE_ARM in paired:
            lkc_diffs = paired[CANDIDATE_ARM]["differences"]
            pooled_lkc_diffs.extend(lkc_diffs)
            task_summary["power_analysis"] = power_analysis(lkc_diffs)
        summary["tasks"][task] = task_summary
    summary["pooled_power_analysis"] = power_analysis(pooled_lkc_diffs)
    summary["pooled_lkc_vs_envelope"] = (
        _mean_std(pooled_lkc_diffs) if pooled_lkc_diffs else None)
    return summary


def _markdown(summary: dict[str, Any]) -> str:
    lines = ["# V19 P2 development-grid summary", ""]
    for task, task_summary in summary["tasks"].items():
        lines += [f"## {task}", "",
                  "| arm | xi-probe (mean +- std) | integrator floor | chance |",
                  "|---|---|---|---|"]
        for arm, cell in task_summary["arms"].items():
            xi, floor = cell["xi_probe"], cell["integrator_floor"]
            lines.append(
                f"| {arm} | {xi['mean']:.4f} +- {xi['std']:.4f} (n={xi['n']}) "
                f"| {floor['mean']:.4f} +- {floor['std']:.4f} "
                f"| {cell['chance']:.3f} |")
        lines.append("")
        paired = task_summary.get("paired_vs_envelope", {})
        if paired:
            lines += ["| arm vs envelope | mean diff | std | wins/seeds |",
                      "|---|---|---|---|"]
            for arm, cell in paired.items():
                lines.append(f"| {arm} | {cell['mean']:+.4f} | {cell['std']:.4f} "
                             f"| {cell['wins']}/{cell['n']} |")
            lines.append("")
        power = task_summary.get("power_analysis")
        if power and power.get("status") == "ok":
            for label, effect in power["effects"].items():
                lines.append(
                    f"- power ({label}, effect={effect['effect']:+.4f}): "
                    + ", ".join(f"n={n}: {p:.2f}"
                                for n, p in effect["power_by_n_seeds"].items())
                    + f" -> smallest n >= 80%: "
                      f"{effect['smallest_n_with_80pct_power']}")
            lines.append("")
    pooled = summary.get("pooled_power_analysis", {})
    if pooled.get("status") == "ok":
        lines += ["## Pooled power analysis (LKC vs recurrent envelope)", ""]
        lines.append(f"- observed effect {pooled['observed_effect']:+.4f} "
                     f"(sd {pooled['observed_sd']:.4f}, "
                     f"n={pooled['n_pairs']} task-seed pairs)")
        for label, effect in pooled["effects"].items():
            lines.append(
                f"- {label}: "
                + ", ".join(f"n={n}: {p:.2f}"
                            for n, p in effect["power_by_n_seeds"].items())
                + f" -> smallest n >= 80%: "
                  f"{effect['smallest_n_with_80pct_power']}")
        lines.append("")
    return "\n".join(lines)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/v19_p2")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    summary = aggregate(args.root)
    root = Path(args.root)
    (root / "p2_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True))
    (root / "p2_summary.md").write_text(_markdown(summary))
    print(f"[v19-p2-aggregate] wrote {root / 'p2_summary.json'} and "
          f"{root / 'p2_summary.md'}", flush=True)


if __name__ == "__main__":
    main()
