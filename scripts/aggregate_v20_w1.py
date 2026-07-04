#!/usr/bin/env python3
"""Aggregate the V20 W1 development grid (claims 4-5 dev readouts + the W3
power analysis; docs/V20_PROPOSAL.md 4.5/6).

Reads every probe_results.json under the W1 root, then:

- arms table per task (registered probe, mean +/- sd over seeds);
- rho* selection for the derived-gain DFC (max pooled mean over dfc_rho*),
  eta* for the fixed-eta control (max pooled mean over dfc_eta*);
- the W1 subsumption gate (claim 4, dev form): paired per-(task, seed)
  differences dfc(rho*) - lkc_rfix on the stationary dev streams must not
  lose — pooled mean >= -0.01 (probe-noise tolerance, registered here);
  a loss falsifies the routing claim before any drift experiment runs;
- eta-localization sanity: mean deployment eta_t of dfc(rho*) on the
  stationary stream (routing predicts ~0: nothing to adapt to);
- the W3 power analysis on the dfc(rho*) - acgru differences (the claim-6
  dichotomy coordinate), via scripts.aggregate_v19_p2.power_analysis.

Writes w1_summary.{json,md}.
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

import scripts.aggregate_v19_p2 as p2agg

TASKS = ("t1dev", "t3dev")
SEEDS = (0, 1, 2)
TRAINED_ARMS = ("none", "acgru", "lkc_rfix")
RHO_VARIANTS = ("dfc_rho6", "dfc_rho4", "dfc_rho2")
ETA_VARIANTS = ("dfc_eta3", "dfc_eta2", "dfc_eta1")
SUBSUMPTION_TOLERANCE = -0.01


def load_probe(root: Path, task: str, arm: str, seed: int) -> dict | None:
    path = root / task / arm / f"s{seed}" / "probe_results.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def registered_score(probe: dict) -> float:
    return float(probe["registered"]["mean"])


def arm_table(root: Path) -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        rows: dict[str, Any] = {}
        for arm in (*TRAINED_ARMS, *RHO_VARIANTS, *ETA_VARIANTS):
            scores = []
            for seed in SEEDS:
                probe = load_probe(root, task, arm, seed)
                if probe is not None:
                    scores.append(registered_score(probe))
            if scores:
                rows[arm] = {
                    "n": len(scores),
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "scores": [round(score, 4) for score in scores],
                }
        table[task] = rows
    return table


def _pooled_mean(table: dict[str, dict[str, Any]], arm: str) -> float | None:
    scores: list[float] = []
    for task in TASKS:
        row = table[task].get(arm)
        if row is None or row["n"] < len(SEEDS):
            return None
        scores.extend(row["scores"])
    return float(np.mean(scores))


def select_variant(table: dict[str, dict[str, Any]],
                   variants: Iterable[str]) -> str | None:
    pooled = {variant: _pooled_mean(table, variant) for variant in variants}
    pooled = {variant: value for variant, value in pooled.items()
              if value is not None}
    if not pooled:
        return None
    return max(pooled, key=pooled.__getitem__)


def paired_differences(root: Path, arm_a: str, arm_b: str
                       ) -> dict[str, Any]:
    """Per-(task, seed) paired differences registered(arm_a) - (arm_b)."""
    diffs: list[float] = []
    per_task: dict[str, list[float]] = {}
    for task in TASKS:
        for seed in SEEDS:
            probe_a = load_probe(root, task, arm_a, seed)
            probe_b = load_probe(root, task, arm_b, seed)
            if probe_a is None or probe_b is None:
                continue
            diff = registered_score(probe_a) - registered_score(probe_b)
            diffs.append(diff)
            per_task.setdefault(task, []).append(diff)
    return {
        "pairs": len(diffs),
        "pooled_mean": float(np.mean(diffs)) if diffs else None,
        "wins": int(sum(diff > 0 for diff in diffs)),
        "per_task_mean": {task: float(np.mean(values))
                          for task, values in per_task.items()},
        "diffs": [round(diff, 4) for diff in diffs],
    }


def stationary_eta(root: Path, variant: str) -> dict[str, Any]:
    """Deployment-gain telemetry of a DFC variant on the stationary stream."""
    etas, drifts = [], []
    for task in TASKS:
        for seed in SEEDS:
            path = root / task / variant / f"s{seed}" / "eval_export.npz"
            if not path.exists():
                continue
            with np.load(path) as data:
                etas.append(float(data["tel_eta_mean"][:, 1:].mean()))
                drifts.append(float(data["phi_episode_end_drift"][-1]))
    return {
        "mean_eta": float(np.mean(etas)) if etas else None,
        "final_phi_drift": float(np.mean(drifts)) if drifts else None,
        "cells": len(etas),
    }


def build_summary(root: Path) -> dict[str, Any]:
    table = arm_table(root)
    rho_star = select_variant(table, RHO_VARIANTS)
    eta_star = select_variant(table, ETA_VARIANTS)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "study": "v20-w1-development-grid",
        "tasks": list(TASKS),
        "seeds": list(SEEDS),
        "arms": table,
        "rho_star": rho_star,
        "eta_star": eta_star,
    }
    if rho_star is not None:
        subsumption = paired_differences(root, rho_star, "lkc_rfix")
        subsumption["tolerance"] = SUBSUMPTION_TOLERANCE
        subsumption["pass"] = bool(
            subsumption["pooled_mean"] is not None
            and subsumption["pooled_mean"] >= SUBSUMPTION_TOLERANCE)
        summary["claim4_subsumption"] = subsumption
        summary["stationary_eta"] = stationary_eta(root, rho_star)
        summary["dfc_vs_etafix"] = (
            paired_differences(root, rho_star, eta_star)
            if eta_star is not None else None)
        dichotomy = paired_differences(root, rho_star, "acgru")
        summary["dfc_vs_acgru_dev"] = dichotomy
        if dichotomy["diffs"]:
            summary["w3_power_analysis"] = p2agg.power_analysis(
                [float(diff) for diff in dichotomy["diffs"]])
    return summary


def render_markdown(summary: dict[str, Any]) -> str:
    lines = ["# V20 W1 summary — development grid", ""]
    for task, rows in summary["arms"].items():
        lines.append(f"## {task} (registered probe, mean ± sd over seeds)")
        lines.append("")
        lines.append("| arm | mean | sd | scores |")
        lines.append("|---|---|---|---|")
        for arm, row in sorted(rows.items(),
                               key=lambda item: -item[1]["mean"]):
            lines.append(f"| {arm} | {row['mean']:.4f} | {row['std']:.4f} | "
                         f"{row['scores']} |")
        lines.append("")
    lines.append(f"**rho\\*** = `{summary['rho_star']}` · "
                 f"**eta\\*** = `{summary['eta_star']}`")
    lines.append("")
    subsumption = summary.get("claim4_subsumption")
    if subsumption:
        lines.append("## Claim 4 — subsumption gate (stationary dev)")
        lines.append("")
        lines.append(f"- paired dfc(rho\\*) − lkc_rfix: pooled mean "
                     f"{subsumption['pooled_mean']:+.4f} "
                     f"({subsumption['wins']}/{subsumption['pairs']} wins), "
                     f"tolerance {subsumption['tolerance']}")
        lines.append(f"- **{'PASS' if subsumption['pass'] else 'FAIL'}**")
        eta_info = summary.get("stationary_eta") or {}
        lines.append(f"- stationary deployment gain: mean eta = "
                     f"{eta_info.get('mean_eta')}, final phi drift = "
                     f"{eta_info.get('final_phi_drift')}")
        lines.append("")
    versus = summary.get("dfc_vs_etafix")
    if versus:
        lines.append(f"## Claim 5 preview — dfc(rho\\*) vs fixed-eta "
                     f"({summary['eta_star']}), stationary")
        lines.append("")
        lines.append(f"- pooled mean {versus['pooled_mean']:+.4f} "
                     f"({versus['wins']}/{versus['pairs']} wins) — the real "
                     f"claim-5 test is W2's drift protocol")
        lines.append("")
    dichotomy = summary.get("dfc_vs_acgru_dev")
    if dichotomy:
        lines.append("## Claim 6 coordinate (dev preview) — dfc(rho\\*) vs acgru")
        lines.append("")
        lines.append(f"- pooled mean {dichotomy['pooled_mean']:+.4f} "
                     f"({dichotomy['wins']}/{dichotomy['pairs']} wins)")
        power = summary.get("w3_power_analysis")
        if power and power.get("status") == "ok":
            for effect_key, effect in power.get("effects", {}).items():
                lines.append(
                    f"- power ({effect_key}): smallest n with >= 80% power = "
                    f"{effect.get('smallest_n_with_80pct_power')}")
        lines.append("")
    return "\n".join(lines)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/v20_w1")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    root = Path(args.root)
    summary = build_summary(root)
    (root / "w1_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (root / "w1_summary.md").write_text(render_markdown(summary))
    print(json.dumps({key: summary.get(key) for key in
                      ("rho_star", "eta_star")}, indent=2))
    subsumption = summary.get("claim4_subsumption")
    if subsumption:
        print(f"claim4 subsumption: mean={subsumption['pooled_mean']} "
              f"pass={subsumption['pass']}")
    print(f"[v20-w1-aggregate] wrote {root / 'w1_summary.md'}")


if __name__ == "__main__":
    main()
