#!/usr/bin/env python3
"""Aggregate the V20 W2 drift protocol (claims 3 and 5; docs/V20_PROPOSAL.md
4.5/6).

Reads the pre/post-shift probe results written by scripts/eval_v20_w2.py and
computes, per drift regime:

- the arms table (pre-shift, post-shift, drift cost);
- claim 5 (adaptation value under drift): paired post-shift differences
  dfc - lkc_rfix and dfc - dfc_etafix, pooled over (task, seed);
- the stationary guard (the W1 subsumption clause re-checked on fresh banks):
  dfc - lkc_rfix on the stationary regime must be >= -0.01;
- eta localization (routing telemetry): post-shift phi velocity vs pre-shift
  phi velocity on drift regimes, and the same ratio on the stationary control
  (prediction: >> 1 under drift, ~1 stationary);
- claim 3 (calibration): dfc's post-shift calibration ratio must sit closer
  to 1.0 than lkc_rfix's on the drift regimes.

Writes w2_summary.{json,md}.
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

REGIMES = ("stationary", "drift_gap", "drift_noise")
TASKS = ("t1dev", "t3dev")      # overridden by --tasks (W3 reuses this module)
SEEDS = (0, 1, 2)               # overridden by --seeds
ARMS = ("dfc", "dfc_etafix", "lkc_rfix", "acgru", "none",
        # exploratory drift arms (registered pre-W2; absent dirs are skipped)
        "dfc_rho4", "dfc_rho2")
STATIONARY_TOLERANCE = -0.01


def load_probe(root: Path, task: str, regime: str, arm: str, seed: int
               ) -> dict | None:
    path = root / task / regime / arm / f"s{seed}" / "probe_results.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def arm_table(root: Path) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for regime in REGIMES:
        regime_rows: dict[str, Any] = {}
        for arm in ARMS:
            pre, post = [], []
            for task in TASKS:
                for seed in SEEDS:
                    probe = load_probe(root, task, regime, arm, seed)
                    if probe is None:
                        continue
                    pre.append(float(probe["pre_shift"]["mean"]))
                    post.append(float(probe["post_shift"]["mean"]))
            if pre:
                regime_rows[arm] = {
                    "n": len(pre),
                    "pre_mean": float(np.mean(pre)),
                    "post_mean": float(np.mean(post)),
                    "drift_cost": float(np.mean(pre) - np.mean(post)),
                }
        table[regime] = regime_rows
    return table


def paired_post(root: Path, regime: str, arm_a: str, arm_b: str
                ) -> dict[str, Any]:
    diffs = []
    for task in TASKS:
        for seed in SEEDS:
            probe_a = load_probe(root, task, regime, arm_a, seed)
            probe_b = load_probe(root, task, regime, arm_b, seed)
            if probe_a is None or probe_b is None:
                continue
            diffs.append(float(probe_a["post_shift"]["mean"])
                         - float(probe_b["post_shift"]["mean"]))
    return {
        "pairs": len(diffs),
        "pooled_mean": float(np.mean(diffs)) if diffs else None,
        "wins": int(sum(diff > 0 for diff in diffs)),
        "diffs": [round(diff, 4) for diff in diffs],
    }


def telemetry_summary(root: Path, arm: str) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for regime in REGIMES:
        ratios, pre_v, post_v = [], [], []
        calib_pre, calib_post = [], []
        for task in TASKS:
            for seed in SEEDS:
                probe = load_probe(root, task, regime, arm, seed)
                if probe is None or "telemetry" not in probe:
                    continue
                telemetry = probe["telemetry"]
                pre_velocity = max(telemetry["pre_shift_phi_velocity"], 1e-12)
                post_v.append(telemetry["post_shift_phi_velocity"])
                pre_v.append(telemetry["pre_shift_phi_velocity"])
                ratios.append(telemetry["post_shift_phi_velocity"]
                              / pre_velocity)
                if "calibration" in probe:
                    calib_pre.append(probe["calibration"]["pre_shift_ratio"])
                    calib_post.append(probe["calibration"]["post_shift_ratio"])
        if ratios:
            rows[regime] = {
                "n": len(ratios),
                "phi_velocity_pre": float(np.mean(pre_v)),
                "phi_velocity_post": float(np.mean(post_v)),
                "velocity_ratio_mean": float(np.mean(ratios)),
                "calibration_pre": (float(np.mean(calib_pre))
                                    if calib_pre else None),
                "calibration_post": (float(np.mean(calib_post))
                                     if calib_post else None),
            }
    return rows


def calibration_claim3(root: Path) -> dict[str, Any]:
    """dfc's post-shift calibration ratio must be closer to 1 than rfix's on
    each drift regime (majority of paired cells)."""
    verdicts: dict[str, Any] = {}
    for regime in ("drift_gap", "drift_noise"):
        closer = 0
        pairs = 0
        details = []
        for task in TASKS:
            for seed in SEEDS:
                probe_dfc = load_probe(root, task, regime, "dfc", seed)
                probe_rfix = load_probe(root, task, regime, "lkc_rfix", seed)
                if (probe_dfc is None or probe_rfix is None
                        or "calibration" not in probe_dfc
                        or "calibration" not in probe_rfix):
                    continue
                pairs += 1
                dfc_gap = abs(probe_dfc["calibration"]["post_shift_ratio"]
                              - 1.0)
                rfix_gap = abs(probe_rfix["calibration"]["post_shift_ratio"]
                               - 1.0)
                closer += dfc_gap < rfix_gap
                details.append({
                    "task": task, "seed": seed,
                    "dfc_post": round(
                        probe_dfc["calibration"]["post_shift_ratio"], 4),
                    "rfix_post": round(
                        probe_rfix["calibration"]["post_shift_ratio"], 4),
                })
        verdicts[regime] = {
            "pairs": pairs,
            "dfc_closer_to_1": closer,
            "pass": bool(pairs and closer * 2 > pairs),
            "details": details,
        }
    return verdicts


def build_summary(root: Path) -> dict[str, Any]:
    table = arm_table(root)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "study": "v20-w2-drift-protocol",
        "regimes": list(REGIMES),
        "tasks": list(TASKS),
        "seeds": list(SEEDS),
        "arms": table,
    }
    stationary_guard = paired_post(root, "stationary", "dfc", "lkc_rfix")
    stationary_guard["tolerance"] = STATIONARY_TOLERANCE
    stationary_guard["pass"] = bool(
        stationary_guard["pooled_mean"] is not None
        and stationary_guard["pooled_mean"] >= STATIONARY_TOLERANCE)
    summary["stationary_guard"] = stationary_guard

    claim5: dict[str, Any] = {}
    for regime in ("drift_gap", "drift_noise"):
        claim5[regime] = {
            "dfc_vs_rfix": paired_post(root, regime, "dfc", "lkc_rfix"),
            "dfc_vs_etafix": paired_post(root, regime, "dfc", "dfc_etafix"),
        }
    summary["claim5_drift"] = claim5

    summary["dfc_telemetry"] = telemetry_summary(root, "dfc")
    summary["claim3_calibration"] = calibration_claim3(root)
    return summary


def render_markdown(summary: dict[str, Any]) -> str:
    lines = ["# V20 W2 summary — drift protocol", ""]
    for regime, rows in summary["arms"].items():
        lines.append(f"## {regime}")
        lines.append("")
        lines.append("| arm | pre-shift | post-shift | drift cost |")
        lines.append("|---|---|---|---|")
        for arm, row in sorted(rows.items(),
                               key=lambda item: -item[1]["post_mean"]):
            lines.append(f"| {arm} | {row['pre_mean']:.4f} | "
                         f"{row['post_mean']:.4f} | "
                         f"{row['drift_cost']:+.4f} |")
        lines.append("")
    guard = summary["stationary_guard"]
    lines.append(f"**Stationary guard** (dfc − rfix, fresh banks): "
                 f"{guard['pooled_mean']:+.4f} "
                 f"({guard['wins']}/{guard['pairs']} wins) — "
                 f"**{'PASS' if guard['pass'] else 'FAIL'}**")
    lines.append("")
    lines.append("## Claim 5 — adaptation value under drift (post-shift)")
    lines.append("")
    for regime, entry in summary["claim5_drift"].items():
        rfix = entry["dfc_vs_rfix"]
        etafix = entry["dfc_vs_etafix"]
        lines.append(f"- **{regime}**: dfc − rfix = {rfix['pooled_mean']:+.4f} "
                     f"({rfix['wins']}/{rfix['pairs']}); dfc − etafix = "
                     f"{etafix['pooled_mean']:+.4f} "
                     f"({etafix['wins']}/{etafix['pairs']})")
    lines.append("")
    lines.append("## Routing telemetry (dfc)")
    lines.append("")
    lines.append("| regime | phi velocity pre | post | ratio | "
                 "calib pre | calib post |")
    lines.append("|---|---|---|---|---|---|")
    for regime, row in summary["dfc_telemetry"].items():
        calib_pre = (f"{row['calibration_pre']:.3f}"
                     if row["calibration_pre"] is not None else "—")
        calib_post = (f"{row['calibration_post']:.3f}"
                      if row["calibration_post"] is not None else "—")
        lines.append(f"| {regime} | {row['phi_velocity_pre']:.2e} | "
                     f"{row['phi_velocity_post']:.2e} | "
                     f"{row['velocity_ratio_mean']:.2f} | "
                     f"{calib_pre} | {calib_post} |")
    lines.append("")
    lines.append("## Claim 3 — calibration restored by adaptation")
    lines.append("")
    for regime, verdict in summary["claim3_calibration"].items():
        lines.append(f"- **{regime}**: dfc closer to 1.0 in "
                     f"{verdict['dfc_closer_to_1']}/{verdict['pairs']} cells "
                     f"— **{'PASS' if verdict['pass'] else 'FAIL'}**")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/v20_w2")
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    global TASKS, SEEDS
    args = parse_args(argv)
    TASKS = tuple(name.strip() for name in args.tasks.split(",")
                  if name.strip())
    SEEDS = tuple(int(value) for value in args.seeds.split(","))
    root = Path(args.root)
    summary = build_summary(root)
    (root / "w2_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (root / "w2_summary.md").write_text(render_markdown(summary))
    print(f"[v20-w2-aggregate] wrote {root / 'w2_summary.md'}")


if __name__ == "__main__":
    main()
