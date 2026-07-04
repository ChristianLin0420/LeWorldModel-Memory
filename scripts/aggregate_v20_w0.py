#!/usr/bin/env python3
"""Aggregate the V20 W0 host preflight + salience ladder (claims 1-2).

Reads every gates.json under the W0 root (and the frozen P0-a2 vicreg
references for t1/t3/t4), selects the registered lambda* (all-seed gate pass
on t1, then max mean final effective rank), reads the ladder certificates
written by scripts/certify_v20_w0.py, computes the per-host certified
salience threshold s* (lowest ladder level whose sighted certificate passes
on >= 2/3 seeds), and writes w0_summary.{json,md}.

Claim verdicts (docs/V20_PROPOSAL.md 5):
  claim 1 PASS  <=> some visreg lambda passes all health gates on 3/3 t1
                    seeds AND lambda* passes 3/3 on t3 and t4.
  claim 2 PASS  <=> s*(visreg lambda*) < s*(vicreg) on the ladder order
                    t1s1 < t1s2 < t1s3 < t1.
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

from lewm.tasks_v19 import make_task

VISREG_ARMS = ("visreg60", "visreg75", "visreg90")
SWEEP_TASK = "t1"
CONFIRM_TASKS = ("t1", "t3", "t4")
LADDER = ("t1s1", "t1s2", "t1s3", "t1")
SEEDS = (0, 1, 2)
MAJORITY = 2
P0A2_ROOT = "outputs/v19_p0_a2"


def load_gates(root: Path, task: str, arm: str, seed: int) -> dict | None:
    path = root / task / arm / f"s{seed}" / "gates.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_certificate(root: Path, task: str, arm: str, seed: int) -> dict | None:
    path = root / "certificates" / task / arm / f"s{seed}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def sweep_table(root: Path) -> dict[str, Any]:
    """Per-lambda health summary on the sweep task."""
    table: dict[str, Any] = {}
    for arm in VISREG_ARMS:
        cells = [load_gates(root, SWEEP_TASK, arm, seed) for seed in SEEDS]
        present = [cell for cell in cells if cell is not None]
        table[arm] = {
            "cells": len(present),
            "passes": sum(bool(cell["overall_pass"]) for cell in present),
            "final_rank": [round(cell["final_effective_rank"], 1)
                           for cell in present],
            "mean_rank": (float(np.mean([cell["final_effective_rank"]
                                         for cell in present]))
                          if present else None),
            "final_ep_ratio": [cell.get("final_ep_ratio") for cell in present],
            "convergence": [round(cell["convergence_relative_change"], 4)
                            for cell in present],
        }
    return table


def select_lambda_star(root: Path) -> str | None:
    """Registered rule: all-3-seed pass on t1, then max mean final rank."""
    table = sweep_table(root)
    passing = {arm: row for arm, row in table.items()
               if row["cells"] == len(SEEDS) and row["passes"] == len(SEEDS)}
    if not passing:
        return None
    return max(passing, key=lambda arm: passing[arm]["mean_rank"])


def construction_salience(task_name: str, episodes: int = 16,
                          seed: int = 20260704) -> float:
    """Mean absolute cue-window pixel difference between paired xi branches —
    the physical x-axis of the s* curve (construction-side, no training)."""
    task = make_task(task_name)
    branch_a, branch_b = task.paired_branches(episodes, seed)
    diffs = []
    for episode in range(episodes):
        on = int(branch_a.events["cue_on"][episode])
        off = int(branch_a.events["cue_off"][episode])
        delta = (branch_a.frames[episode, on:off].astype(np.float64)
                 - branch_b.frames[episode, on:off].astype(np.float64))
        diffs.append(np.abs(delta).mean())
    return float(np.mean(diffs))


def ladder_readout(root: Path, arm: str) -> dict[str, Any]:
    """Per-level sighted results and the s* threshold for one arm."""
    levels: dict[str, Any] = {}
    s_star: str | None = None
    for level in LADDER:
        certificates = [load_certificate(root, level, arm, seed)
                        for seed in SEEDS]
        present = [cert for cert in certificates if cert is not None]
        sighted = [cert["sighted"] for cert in present]
        passes = sum(bool(entry["pass"]) for entry in sighted)
        level_pass = len(present) == len(SEEDS) and passes >= MAJORITY
        levels[level] = {
            "cells": len(present),
            "sighted_scores": [round(entry["score"], 3) for entry in sighted],
            "sighted_passes": passes,
            "integrator_passes": sum(bool(cert["integrator"]["pass"])
                                     for cert in present),
            "level_pass": bool(level_pass),
        }
        if level_pass and s_star is None:
            s_star = level
    return {"levels": levels, "s_star": s_star,
            "s_star_index": (LADDER.index(s_star) if s_star is not None
                             else len(LADDER))}


def health_table(root: Path, arm: str, reference_root: Path) -> dict[str, Any]:
    """lambda* health on the confirmation tasks vs the frozen P0-a2 vicreg."""
    table: dict[str, Any] = {}
    for task in CONFIRM_TASKS:
        cells = [load_gates(root, task, arm, seed) for seed in SEEDS]
        present = [cell for cell in cells if cell is not None]
        reference = []
        for seed in SEEDS:
            path = reference_root / task / "vicreg" / f"s{seed}" / "gates.json"
            if path.exists():
                reference.append(json.loads(path.read_text()))
        table[task] = {
            "cells": len(present),
            "passes": sum(bool(cell["overall_pass"]) for cell in present),
            "final_rank": [round(cell["final_effective_rank"], 1)
                           for cell in present],
            "vicreg_reference_rank": [
                round(cell["final_effective_rank"], 1) for cell in reference],
            "vicreg_reference_passes": sum(bool(cell["overall_pass"])
                                           for cell in reference),
        }
    return table


def build_summary(root: Path) -> dict[str, Any]:
    lambda_star = select_lambda_star(root)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "study": "v20-w0-host-preflight-and-salience-ladder",
        "sweep_task": SWEEP_TASK,
        "ladder": list(LADDER),
        "seeds": list(SEEDS),
        "lambda_sweep": sweep_table(root),
        "lambda_star": lambda_star,
        "construction_salience": {level: round(construction_salience(level), 3)
                                  for level in LADDER},
    }
    claim1 = False
    if lambda_star is not None:
        health = health_table(root, lambda_star, Path(P0A2_ROOT))
        summary["lambda_star_health"] = health
        claim1 = all(row["cells"] == len(SEEDS)
                     and row["passes"] == len(SEEDS)
                     for row in health.values())
    summary["claim1_visreg_host_healthy"] = bool(claim1)

    readouts: dict[str, Any] = {"vicreg": ladder_readout(root, "vicreg")}
    if lambda_star is not None:
        readouts[lambda_star] = ladder_readout(root, lambda_star)
    summary["ladder_readout"] = readouts

    claim2 = None
    if lambda_star is not None:
        visreg_index = readouts[lambda_star]["s_star_index"]
        vicreg_index = readouts["vicreg"]["s_star_index"]
        claim2 = bool(visreg_index < vicreg_index)
        summary["s_star"] = {
            "visreg": readouts[lambda_star]["s_star"],
            "vicreg": readouts["vicreg"]["s_star"],
        }
    summary["claim2_visreg_lowers_s_star"] = claim2
    return summary


def render_markdown(summary: dict[str, Any]) -> str:
    lines = ["# V20 W0 summary — host preflight + salience ladder", ""]
    lines.append("## Lambda sweep (t1, health gates)")
    lines.append("")
    lines.append("| arm | passes | final rank (per seed) | mean rank | "
                 "final EP ratio |")
    lines.append("|---|---|---|---|---|")
    for arm, row in summary["lambda_sweep"].items():
        mean_rank = (f"{row['mean_rank']:.1f}"
                     if row["mean_rank"] is not None else "—")
        ep_values = ", ".join(
            f"{value:.3f}" if isinstance(value, float) else "—"
            for value in row["final_ep_ratio"])
        lines.append(f"| {arm} | {row['passes']}/{row['cells']} | "
                     f"{row['final_rank']} | {mean_rank} | {ep_values} |")
    lines.append("")
    lines.append(f"**lambda\\*** = `{summary['lambda_star']}` · "
                 f"**claim 1 (host healthy)**: "
                 f"{'PASS' if summary['claim1_visreg_host_healthy'] else 'FAIL'}")
    lines.append("")
    if "lambda_star_health" in summary:
        lines.append("## lambda\\* health on the confirmation tasks")
        lines.append("")
        lines.append("| task | passes | final rank | vicreg reference rank |")
        lines.append("|---|---|---|---|")
        for task, row in summary["lambda_star_health"].items():
            lines.append(f"| {task} | {row['passes']}/{row['cells']} | "
                         f"{row['final_rank']} | "
                         f"{row['vicreg_reference_rank']} |")
        lines.append("")
    lines.append("## Salience ladder (sighted certificate per level)")
    lines.append("")
    salience = summary["construction_salience"]
    for arm, readout in summary["ladder_readout"].items():
        lines.append(f"### {arm}")
        lines.append("")
        lines.append("| level | construction salience | sighted scores | "
                     "passes | level pass |")
        lines.append("|---|---|---|---|---|")
        for level, row in readout["levels"].items():
            lines.append(
                f"| {level} | {salience[level]:.3f} | "
                f"{row['sighted_scores']} | "
                f"{row['sighted_passes']}/{row['cells']} | "
                f"{'PASS' if row['level_pass'] else 'fail'} |")
        lines.append("")
        lines.append(f"**s\\*({arm})** = `{readout['s_star']}`")
        lines.append("")
    claim2 = summary["claim2_visreg_lowers_s_star"]
    verdict = ("PASS" if claim2 else
               "FAIL" if claim2 is not None else "not evaluable")
    lines.append(f"**claim 2 (VisReg lowers s\\*)**: {verdict}")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/v20_w0")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    root = Path(args.root)
    summary = build_summary(root)
    (root / "w0_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (root / "w0_summary.md").write_text(render_markdown(summary))
    print(json.dumps({key: summary[key] for key in
                      ("lambda_star", "claim1_visreg_host_healthy",
                       "s_star" if "s_star" in summary else "ladder",
                       "claim2_visreg_lowers_s_star")
                      if key in summary}, indent=2))
    print(f"[v20-w0-aggregate] wrote {root / 'w0_summary.md'}")


if __name__ == "__main__":
    main()
