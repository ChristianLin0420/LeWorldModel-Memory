#!/usr/bin/env python3
"""Aggregate V19 P1b checkpoint certificates into p1b_summary.{json,md}.

Reads every ``certificate.json`` written by scripts/certify_v19_p1b.py under
the P1b root and reports, per task x host: integrator scores vs their
permutation-null thresholds, sighted scores vs their gates, probe-level
memory demand, truncation curves, and the two-sided pass (a task-encoder
pair is certified only when every seed passes both sides).
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


def collect(root: str | Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """{task: {host: [certificates sorted by seed]}}."""
    table: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for path in sorted(Path(root).glob("*/*/s*/certificate.json")):
        certificate = json.loads(path.read_text())
        table.setdefault(certificate["task"], {}).setdefault(
            certificate["host"], []).append(certificate)
    if not table:
        raise FileNotFoundError(f"no certificate.json files under {root}")
    for hosts in table.values():
        for certificates in hosts.values():
            certificates.sort(key=lambda cert: cert["seed"])
    return table


def _cell_summary(certificates: list[dict[str, Any]]) -> dict[str, Any]:
    integrator = [cert["integrator"]["score"] for cert in certificates]
    nulls = [cert["integrator"]["null_95pct"] for cert in certificates]
    sighted = [cert["sighted"]["score"] for cert in certificates]
    gates = [cert["sighted"]["gate"] for cert in certificates]
    demand = [cert["memory_demand"] for cert in certificates]
    windows = sorted(certificates[0]["truncation_curve"],
                     key=lambda w: np.inf if w == "full" else int(w))
    curve = {window: float(np.mean(
        [cert["truncation_curve"][window] for cert in certificates]))
        for window in windows}
    return {
        "seeds": [cert["seed"] for cert in certificates],
        "chance": certificates[0]["chance"],
        "xi_kind": certificates[0]["xi_kind"],
        "integrator_scores": integrator,
        "integrator_null_95pct": nulls,
        "integrator_pass_count": sum(
            cert["integrator"]["pass"] for cert in certificates),
        "sighted_scores": sighted,
        "sighted_gates": gates,
        "sighted_pass_count": sum(
            cert["sighted"]["pass"] for cert in certificates),
        "memory_demand_mean": float(np.mean(demand)),
        "memory_demand_std": float(np.std(demand)),
        "truncation_curve_mean": curve,
        "per_seed_two_sided": [cert["two_sided_pass"]
                               for cert in certificates],
        "two_sided_pass": bool(all(cert["two_sided_pass"]
                                   for cert in certificates)),
    }


def aggregate(root: str | Path) -> dict[str, Any]:
    table = collect(root)
    summary: dict[str, Any] = {"schema_version": 1, "root": str(root),
                               "cells": {}}
    for task, hosts in sorted(table.items()):
        for host, certificates in sorted(hosts.items()):
            summary["cells"][f"{task}/{host}"] = _cell_summary(certificates)
    return summary


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# V19 P1b checkpoint-certificate summary", "",
        "| task/host | integrator (vs null95) | sighted (vs gate) "
        "| memory demand | two-sided |",
        "|---|---|---|---|---|",
    ]
    for cell, data in summary["cells"].items():
        integrator = ", ".join(
            f"{score:.3f}<={null:.3f}" for score, null in
            zip(data["integrator_scores"], data["integrator_null_95pct"]))
        sighted = ", ".join(
            f"{score:.3f}>={gate:.3f}" for score, gate in
            zip(data["sighted_scores"], data["sighted_gates"]))
        verdict = "PASS" if data["two_sided_pass"] else "FAIL"
        lines.append(
            f"| {cell} | {integrator} "
            f"({data['integrator_pass_count']}/{len(data['seeds'])}) "
            f"| {sighted} ({data['sighted_pass_count']}/{len(data['seeds'])}) "
            f"| {data['memory_demand_mean']:.3f} "
            f"+- {data['memory_demand_std']:.3f} | {verdict} |")
    lines += ["", "## Truncation curves (seed means)", ""]
    for cell, data in summary["cells"].items():
        curve = ", ".join(f"w={window}: {value:.3f}"
                          for window, value in
                          data["truncation_curve_mean"].items())
        lines.append(f"- {cell}: {curve}")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/v19_p1b")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    summary = aggregate(args.root)
    root = Path(args.root)
    (root / "p1b_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True))
    (root / "p1b_summary.md").write_text(_markdown(summary))
    print(f"[v19-p1b-aggregate] wrote {root / 'p1b_summary.json'} and "
          f"{root / 'p1b_summary.md'}", flush=True)


if __name__ == "__main__":
    main()
