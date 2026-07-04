#!/usr/bin/env python3
"""V21 X0a — Tier-0 sensitivity analysis of the W3 inversion contrast
(docs/V21_PROPOSAL.md 4/X0.1; panel objection I1/I2 'free analyses').

Recomputes the W3 lkc_rfix - acgru contrast on the frozen probe results
under registered exclusion rules, and publishes per-task seed-level CIs so
the crossed 2-task bootstrap is not the only statistic:

  full          all 10 seeds, t1/t3 (the V20 section-11 coordinate)
  healthy_only  drop any (task, seed) where EITHER arm's Tier-0 gates
                failed (pairwise exclusion; 13/90 cells failed overall)
  no_clusters   drop the shared task-x-seed convergence clusters
                (t1/s3, t1/s5, t3/s3) plus rfix's own t3 failures
                (t3/s2, t3/s7, t3/s8)

Also reports the same three variants for dfc(rho*) - lkc_rfix (claim 4's
per-task honest split), and per-task one-sample t CIs on the seed-paired
differences.  Pure re-analysis: no training, no new probes.

Writes outputs/v21_x0/sensitivity.{json,md}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.gates_v19_p3 import crossed_bootstrap

W3 = ROOT / "outputs" / "v20_w3"
OUT = ROOT / "outputs" / "v21_x0"
TASKS = ("t1", "t3")
SEEDS = tuple(range(10))
CLUSTER_EXCLUSIONS = {("t1", 3), ("t1", 5), ("t3", 3),
                      ("t3", 2), ("t3", 7), ("t3", 8)}


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def probe_score(task: str, arm: str, seed: int) -> float:
    return float(_load(W3 / task / arm / f"s{seed}" / "probe_results.json")
                 ["registered"]["mean"])


def cell_healthy(task: str, arm: str, seed: int) -> bool:
    return bool(_load(W3 / task / arm / f"s{seed}" / "gates.json")
                ["overall_pass"])


def paired_cells(arm_a: str, arm_b: str, keep) -> dict[str, list[float]]:
    """Per-task lists of paired differences over the kept (task, seed) cells."""
    diffs: dict[str, list[float]] = {}
    for task in TASKS:
        for seed in SEEDS:
            if not keep(task, seed, arm_a, arm_b):
                continue
            diffs.setdefault(task, []).append(
                probe_score(task, arm_a, seed) - probe_score(task, arm_b, seed))
    return diffs


def summarize(diffs: dict[str, list[float]]) -> dict[str, Any]:
    """Per-task t-based CIs + a crossed bootstrap when the grid is full
    (equal seed counts per task); otherwise pooled t on the flat list."""
    per_task: dict[str, Any] = {}
    for task, values in diffs.items():
        array = np.asarray(values, dtype=np.float64)
        se = array.std(ddof=1) / np.sqrt(len(array))
        margin = stats.t.ppf(0.975, df=len(array) - 1) * se
        per_task[task] = {
            "n": len(array),
            "mean": float(array.mean()),
            "ci95_t": [float(array.mean() - margin),
                       float(array.mean() + margin)],
            "wins": int((array > 0).sum()),
        }
    flat = np.concatenate([np.asarray(v) for v in diffs.values()])
    counts = {len(v) for v in diffs.values()}
    if len(counts) == 1 and len(diffs) == len(TASKS):
        matrix = np.stack([np.asarray(diffs[task]) for task in TASKS])
        pooled = crossed_bootstrap(matrix)
        pooled_entry = {"method": "crossed_bootstrap",
                        "mean": pooled["mean"],
                        "ci95": [pooled["ci95_low"], pooled["ci95_high"]],
                        "p_pos": pooled["p_pos"]}
    else:
        se = flat.std(ddof=1) / np.sqrt(len(flat))
        margin = stats.t.ppf(0.975, df=len(flat) - 1) * se
        p = float(stats.ttest_1samp(flat, 0.0, alternative="greater").pvalue)
        pooled_entry = {"method": "flat_t_unequal_cells",
                        "mean": float(flat.mean()),
                        "ci95": [float(flat.mean() - margin),
                                 float(flat.mean() + margin)],
                        "p_pos": p}
    return {"per_task": per_task, "pooled": pooled_entry,
            "total_pairs": int(len(flat)),
            "total_wins": int((flat > 0).sum())}


def main() -> None:
    variants = {
        "full": lambda task, seed, a, b: True,
        "healthy_only": lambda task, seed, a, b: (
            cell_healthy(task, a, seed) and cell_healthy(task, b, seed)),
        "no_clusters": lambda task, seed, a, b: (
            (task, seed) not in CLUSTER_EXCLUSIONS),
    }
    contrasts = {"inversion_rfix_minus_acgru": ("lkc_rfix", "acgru"),
                 "claim4_dfc_minus_rfix": ("dfc_rho6", "lkc_rfix")}
    report: dict[str, Any] = {
        "schema_version": 1,
        "study": "v21-x0-tier0-sensitivity",
        "tasks": list(TASKS),
        "seeds": list(SEEDS),
        "cluster_exclusions": sorted(f"{t}/s{s}"
                                     for t, s in CLUSTER_EXCLUSIONS),
        "note": ("dfc/dfc_eta variants have no own gates.json (deployment "
                 "evals of the lkc_rfix checkpoint); healthy_only uses the "
                 "underlying checkpoint's gates for them"),
    }
    for name, (arm_a, arm_b) in contrasts.items():
        gate_a = "lkc_rfix" if arm_a.startswith("dfc") else arm_a
        gate_b = "lkc_rfix" if arm_b.startswith("dfc") else arm_b
        report[name] = {}
        for variant, rule in variants.items():
            keep = (lambda t, s, _a, _b, r=rule, ga=gate_a, gb=gate_b:
                    r(t, s, ga, gb))
            report[name][variant] = summarize(paired_cells(arm_a, arm_b, keep))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sensitivity.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")

    lines = ["# V21 X0a — Tier-0 sensitivity of the W3 contrasts", ""]
    for name in contrasts:
        lines.append(f"## {name}")
        lines.append("")
        lines.append("| variant | pooled mean | CI95 | p_pos | pairs (wins) | "
                     "t1 mean [CI] | t3 mean [CI] |")
        lines.append("|---|---|---|---|---|---|---|")
        for variant in variants:
            entry = report[name][variant]
            pooled = entry["pooled"]
            def _fmt(task):
                row = entry["per_task"].get(task)
                if row is None:
                    return "—"
                return (f"{row['mean']:+.4f} [{row['ci95_t'][0]:+.3f}, "
                        f"{row['ci95_t'][1]:+.3f}] ({row['wins']}/{row['n']})")
            lines.append(
                f"| {variant} | {pooled['mean']:+.4f} | "
                f"[{pooled['ci95'][0]:+.4f}, {pooled['ci95'][1]:+.4f}] | "
                f"{pooled['p_pos']:.2e} | {entry['total_pairs']} "
                f"({entry['total_wins']}) | {_fmt('t1')} | {_fmt('t3')} |")
        lines.append("")
    (OUT / "sensitivity.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
