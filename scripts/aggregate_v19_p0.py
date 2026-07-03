#!/usr/bin/env python3
"""Aggregate V19 P0 host-preflight gates into one summary JSON + markdown.

Reads ``<root>/<task>/<host>/s<seed>/gates.json`` for every cell of the
2 hosts x 4 tasks x 3 seeds grid, reports per task x host health (final
effective rank mean+-sd over seeds, final channel variance, convergence,
plateau/gradient-ratio flags, gates passed n/3), then applies the registered
per-task attribution rule of docs/V19_PROPOSAL.md section 4.1:

- sigreg fails while vicreg passes  -> 'host fault'
- both fail                         -> 'task fault'
- both pass                         -> 'healthy'
- sigreg passes while vicreg fails  -> 'reference fault (unexpected)'

An arm passes a task only if all three seeds pass their gates (the V18
every-cell discipline); missing cells make the attribution 'incomplete'.
Writes ``p0_summary.json`` and ``p0_summary.md`` under the root.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.make_v19_p0_data import P0_TASKS
from scripts.train_v19_p0 import HOSTS

SEEDS = (0, 1, 2)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/v19_p0")
    return parser.parse_args(argv)


def _load(root: Path, task: str, host: str, seed: int) -> dict | None:
    path = root / task / host / f"s{seed}" / "gates.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _mean_sd(gates: list[dict], key: str) -> str:
    values = [gate[key] for gate in gates if gate.get(key) is not None]
    if not values:
        return "—"
    return f"{np.mean(values):.3g}±{np.std(values):.2g}"


def _flag_cell(gates: list[dict], key: str) -> str:
    if not gates:
        return "—"
    return f"{sum(bool(gate[key]) for gate in gates)}/{len(gates)}"


def _arm_summary(gates_by_seed: dict[int, dict | None]) -> dict:
    gates = [gate for gate in gates_by_seed.values() if gate is not None]
    complete = len(gates) == len(SEEDS)
    return {
        "n_found": len(gates),
        "complete": complete,
        "seeds_pass": sum(bool(gate["overall_pass"]) for gate in gates),
        # Arm-level verdict: every seed must pass (fail-closed, V18 style).
        "arm_pass": complete and all(gate["overall_pass"] for gate in gates),
        "final_effective_rank": [gate["final_effective_rank"] for gate in gates],
        "final_channel_variance": [gate["final_channel_variance"]
                                   for gate in gates],
        "convergence_relative_change": [gate["convergence_relative_change"]
                                        for gate in gates],
        "plateau_flags": sum(bool(gate["plateau_flag"]) for gate in gates),
        "grad_ratio_flags": sum(bool(gate["grad_ratio_flag"]) for gate in gates),
        "per_seed": {str(seed): gate for seed, gate in gates_by_seed.items()},
    }


def _attribution(sigreg: dict, vicreg: dict) -> str:
    if not (sigreg["complete"] and vicreg["complete"]):
        return "incomplete"
    if sigreg["arm_pass"] and vicreg["arm_pass"]:
        return "healthy"
    if not sigreg["arm_pass"] and vicreg["arm_pass"]:
        return "host fault"
    if not sigreg["arm_pass"] and not vicreg["arm_pass"]:
        return "task fault"
    return "reference fault (unexpected)"


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    root = Path(args.root)
    summary: dict = {"root": str(root), "seeds": list(SEEDS),
                     "hosts": list(HOSTS), "tasks": {}}
    lines = [
        "| task | host | rank (final) | variance (final) | convergence | "
        "plateau flag | grad>100 flag | gates | attribution |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for task in P0_TASKS:
        arms = {}
        for host in HOSTS:
            gates_by_seed = {seed: _load(root, task, host, seed)
                             for seed in SEEDS}
            arms[host] = _arm_summary(gates_by_seed)
        attribution = _attribution(arms["sigreg"], arms["vicreg"])
        summary["tasks"][task] = {"arms": arms, "attribution": attribution}
        for index, host in enumerate(HOSTS):
            arm = arms[host]
            gates = [gate for gate in arm["per_seed"].values()
                     if gate is not None]
            convergence = _flag_cell(gates, "convergence_pass")
            if convergence != "—":
                convergence += " pass"
            lines.append(
                f"| {task} | {host} | "
                f"{_mean_sd(gates, 'final_effective_rank')} | "
                f"{_mean_sd(gates, 'final_channel_variance')} | "
                f"{convergence} | "
                f"{_flag_cell(gates, 'plateau_flag')} | "
                f"{_flag_cell(gates, 'grad_ratio_flag')} | "
                f"{arm['seeds_pass']}/{len(SEEDS)} | "
                f"{attribution if index == 0 else ''} |")

    markdown = ("# V19 P0 host-preflight summary\n\n"
                "Arm pass requires all seeds to pass "
                "(rank>=16, variance>=1e-4, convergence<=5%, and for sigreg "
                "no projected-zero plateau).\n\n" + "\n".join(lines) + "\n")
    root.mkdir(parents=True, exist_ok=True)
    (root / "p0_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True))
    (root / "p0_summary.md").write_text(markdown)
    print(markdown, flush=True)


if __name__ == "__main__":
    main()
