#!/usr/bin/env python3
"""V20 W3 frozen-confirmation gates (docs/V20_PROPOSAL.md 4.5/5).

Statistical machinery is scripts/gates_v19_p3.py verbatim (crossed
tasks x seeds bootstrap, add-one p-values, Holm step-down).  The registered
confirmatory family (Holm-corrected together):

  C4  subsumption (stationary, T1/T3/T4): paired dfc(rho*) - lkc_rfix,
      non-inferiority at tolerance -0.01 (matrix shifted by +0.01, p_pos)
  C5g adaptation value under drift_gap  (post-shift, T1/T3): dfc - rfix > 0
  C5n adaptation value under drift_noise (post-shift, T1/T3): dfc - rfix > 0
  C5e derived vs constant gain (post-shift, both drift regimes pooled):
      dfc - dfc_etafix > 0

Estimation (reported with CI, not Holm-tested):

  C6  the dichotomy: T1 gap closure = mean(dfc - rfix) / mean(acgru - rfix)
      on the stationary confirmation streams, seed-bootstrap CI.  >= 0.5
      reads adaptivity; below with mechanism gates green reads nonlinearity.

Mechanism gates (fail-closed report): eta localization (post/pre phi
velocity ratio on drift vs stationary) and the claim-3 calibration verdicts
from the W3 drift aggregation.  Tier-0 health is reported per arm.

Inputs: <root> (stationary grid + probe_results.json), <root>/drift (the
W2-protocol outputs on the frozen tasks), the W1 summary (rho*/eta* variant
names).  Writes w3_gates.{json,md}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.gates_v19_p3 import crossed_bootstrap, holm, wilson_ci
import scripts.aggregate_v20_w2 as w2agg

STATIONARY_TASKS = ("t1", "t3", "t4")
DRIFT_TASKS = ("t1", "t3")
DRIFT_REGIMES = ("drift_gap", "drift_noise")
TOLERANCE = 0.01                 # C4 non-inferiority margin
HOLM_ALPHA = 0.05


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def stationary_probe(root: Path, task: str, arm: str, seed: int
                     ) -> float | None:
    probe = _load(root / task / arm / f"s{seed}" / "probe_results.json")
    return None if probe is None else float(probe["registered"]["mean"])


def drift_probe(root: Path, task: str, regime: str, arm: str, seed: int,
                half: str) -> float | None:
    probe = _load(root / "drift" / task / regime / arm / f"s{seed}"
                  / "probe_results.json")
    return None if probe is None else float(probe[half]["mean"])


def paired_matrix(root: Path, tasks: Iterable[str], seeds: Iterable[int],
                  arm_a: str, arm_b: str) -> np.ndarray:
    rows = []
    for task in tasks:
        row = []
        for seed in seeds:
            score_a = stationary_probe(root, task, arm_a, seed)
            score_b = stationary_probe(root, task, arm_b, seed)
            if score_a is None or score_b is None:
                raise FileNotFoundError(
                    f"missing stationary probes for {task}/"
                    f"{arm_a}|{arm_b}/s{seed}")
            row.append(score_a - score_b)
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)


def drift_matrix(root: Path, regimes: Iterable[str], seeds: Iterable[int],
                 arm_a: str, arm_b: str) -> np.ndarray:
    """(task x regime, seeds) post-shift paired differences."""
    rows = []
    for task in DRIFT_TASKS:
        for regime in regimes:
            row = []
            for seed in seeds:
                score_a = drift_probe(root, task, regime, arm_a, seed,
                                      "post_shift")
                score_b = drift_probe(root, task, regime, arm_b, seed,
                                      "post_shift")
                if score_a is None or score_b is None:
                    raise FileNotFoundError(
                        f"missing drift probes for {task}/{regime}/"
                        f"{arm_a}|{arm_b}/s{seed}")
                row.append(score_a - score_b)
            rows.append(row)
    return np.asarray(rows, dtype=np.float64)


def arm_health(root: Path, arms: Iterable[str], seeds: Iterable[int]
               ) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for arm in arms:
        passes = cells = 0
        for task in STATIONARY_TASKS:
            for seed in seeds:
                gates = _load(root / task / arm / f"s{seed}" / "gates.json")
                if gates is None:
                    continue
                cells += 1
                passes += bool(gates["overall_pass"])
        table[arm] = {"cells": cells, "passes": passes}
    return table


def gap_closure(root: Path, seeds: Iterable[int], dfc_arm: str,
                bootstrap_draws: int = 100_000) -> dict[str, Any]:
    """C6 on T1: mean(dfc - rfix) / mean(acgru - rfix), seed bootstrap."""
    seeds = list(seeds)
    dfc = np.array([stationary_probe(root, "t1", dfc_arm, seed)
                    for seed in seeds], dtype=np.float64)
    rfix = np.array([stationary_probe(root, "t1", "lkc_rfix", seed)
                     for seed in seeds], dtype=np.float64)
    acgru = np.array([stationary_probe(root, "t1", "acgru", seed)
                      for seed in seeds], dtype=np.float64)
    if np.isnan(dfc).any() or np.isnan(rfix).any() or np.isnan(acgru).any():
        raise ValueError("missing T1 probes for gap closure")
    gap = float((acgru - rfix).mean())
    closure = (float((dfc - rfix).mean()) / gap) if abs(gap) > 1e-9 else None
    rng = np.random.default_rng(19_019)
    ratios = []
    for _ in range(bootstrap_draws // 100):
        index = rng.integers(0, len(seeds), size=len(seeds))
        gap_b = (acgru[index] - rfix[index]).mean()
        if abs(gap_b) > 1e-9:
            ratios.append((dfc[index] - rfix[index]).mean() / gap_b)
    ratios = np.asarray(ratios)
    return {
        "t1_envelope_gap_acgru_minus_rfix": gap,
        "t1_dfc_minus_rfix": float((dfc - rfix).mean()),
        "closure": closure,
        "closure_ci95": ([float(np.quantile(ratios, 0.025)),
                          float(np.quantile(ratios, 0.975))]
                         if ratios.size else None),
        "note": ("envelope gap <= 0: rfix already matches the envelope; "
                 "dichotomy moot" if gap <= 0 else None),
    }


def build_gates(root: Path, w1_summary_path: Path, seeds: list[int]
                ) -> dict[str, Any]:
    summary = json.loads(w1_summary_path.read_text())
    dfc_arm = summary["rho_star"]           # stationary variant dir name
    eta_arm = summary["eta_star"]

    tests: dict[str, Any] = {}
    # C4: non-inferiority — shift by +TOLERANCE, test mean > 0.
    c4_matrix = paired_matrix(root, STATIONARY_TASKS, seeds,
                              dfc_arm, "lkc_rfix")
    tests["C4_subsumption"] = {
        "matrix_tasks": list(STATIONARY_TASKS),
        "tolerance": TOLERANCE,
        **crossed_bootstrap(c4_matrix + TOLERANCE),
        "raw_mean": float(c4_matrix.mean()),
    }
    # C5: drift arms use canonical names (dfc/dfc_etafix/lkc_rfix).
    for name, regimes in (("C5_drift_gap", ("drift_gap",)),
                          ("C5_drift_noise", ("drift_noise",))):
        matrix = drift_matrix(root, regimes, seeds, "dfc", "lkc_rfix")
        tests[name] = {"matrix_rows": f"{DRIFT_TASKS} x {regimes}",
                       **crossed_bootstrap(matrix)}
    matrix_eta = drift_matrix(root, DRIFT_REGIMES, seeds, "dfc", "dfc_etafix")
    tests["C5_derived_vs_fixed_eta"] = {
        "matrix_rows": f"{DRIFT_TASKS} x {DRIFT_REGIMES}",
        **crossed_bootstrap(matrix_eta)}

    corrected = holm({name: entry["p_pos"]
                      for name, entry in tests.items()}, HOLM_ALPHA)
    for name, entry in tests.items():
        entry["holm"] = corrected[name]

    gates: dict[str, Any] = {
        "schema_version": 1,
        "study": "v20-w3-frozen-confirmation",
        "seeds": seeds,
        "dfc_variant": dfc_arm,
        "etafix_variant": eta_arm,
        "tier0_health": arm_health(
            root, ("none", "acgru", "lkc_rfix"), seeds),
        "confirmatory": tests,
        "C6_gap_closure": gap_closure(root, seeds, dfc_arm),
    }

    # Mechanism report from the drift aggregation (written separately by
    # aggregate_v20_w2 on <root>/drift).
    drift_summary = _load(root / "drift" / "w2_summary.json")
    if drift_summary is not None:
        gates["mechanism"] = {
            "eta_localization": drift_summary.get("dfc_telemetry"),
            "claim3_calibration": drift_summary.get("claim3_calibration"),
            "stationary_guard": drift_summary.get("stationary_guard"),
        }
    return gates


def render_markdown(gates: dict[str, Any]) -> str:
    lines = ["# V20 W3 gates — frozen confirmation", ""]
    lines.append(f"seeds: {gates['seeds']} · dfc = `{gates['dfc_variant']}` "
                 f"· etafix = `{gates['etafix_variant']}`")
    lines.append("")
    lines.append("## Tier 0 — health")
    lines.append("")
    for arm, row in gates["tier0_health"].items():
        lines.append(f"- {arm}: {row['passes']}/{row['cells']} pass")
    lines.append("")
    lines.append("## Confirmatory family (Holm)")
    lines.append("")
    lines.append("| claim | mean | CI95 | p | p_holm | reject H0 |")
    lines.append("|---|---|---|---|---|---|")
    for name, entry in gates["confirmatory"].items():
        mean = entry.get("raw_mean", entry["mean"])
        lines.append(
            f"| {name} | {mean:+.4f} | [{entry['ci95_low']:+.4f}, "
            f"{entry['ci95_high']:+.4f}] | {entry['holm']['p']:.5f} | "
            f"{entry['holm']['p_holm']:.5f} | "
            f"{'YES' if entry['holm']['reject'] else 'no'} |")
    lines.append("")
    closure = gates["C6_gap_closure"]
    lines.append("## C6 — the dichotomy (T1 gap closure)")
    lines.append("")
    lines.append(f"- envelope gap (acgru − rfix): "
                 f"{closure['t1_envelope_gap_acgru_minus_rfix']:+.4f}")
    lines.append(f"- dfc − rfix: {closure['t1_dfc_minus_rfix']:+.4f}")
    lines.append(f"- closure: {closure['closure']} "
                 f"(CI95 {closure['closure_ci95']})")
    if closure.get("note"):
        lines.append(f"- note: {closure['note']}")
    lines.append("")
    mechanism = gates.get("mechanism")
    if mechanism:
        lines.append("## Mechanism (drift telemetry)")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(
            {key: value for key, value in mechanism.items()
             if key != "eta_localization"}, indent=2, sort_keys=True))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/v20_w3")
    parser.add_argument("--w1-summary", default="outputs/v20_w1/w1_summary.json")
    parser.add_argument("--seeds", required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    root = Path(args.root)
    seeds = [int(value) for value in args.seeds.split(",")]
    gates = build_gates(root, Path(args.w1_summary), seeds)
    (root / "w3_gates.json").write_text(
        json.dumps(gates, indent=2, sort_keys=True) + "\n")
    (root / "w3_gates.md").write_text(render_markdown(gates))
    print(render_markdown(gates))


if __name__ == "__main__":
    main()
