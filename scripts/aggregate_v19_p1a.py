#!/usr/bin/env python3
"""Aggregate V19 P1a certificates into one summary JSON + markdown table.

Reads ``<root>/<task>/s<seed>/certificate.json`` for every registered task and
seed, pools clause values over seeds and both action streams, and writes
``p1a_summary.json`` and ``p1a_summary.md``.  Missing cells degrade to '—' so
the summary is usable while the grid is still running.
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

from lewm.tasks_v19 import TASKS

SEEDS = (0, 1, 2)
STREAMS = ("iid", "script")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/v19_p1a")
    return parser.parse_args(argv)


def _load(root: Path, task: str, seed: int) -> dict | None:
    path = root / task / f"s{seed}" / "certificate.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _pooled(certs: list[dict], clause: str) -> str:
    """mean±sd of one clause value over available (seed, stream) cells."""
    values = [cert["streams"][stream][clause]["value"]
              for cert in certs for stream in STREAMS
              if clause in cert["streams"].get(stream, {})]
    if not values:
        return "—"
    return f"{np.mean(values):.3f}±{np.std(values):.3f}"


def _rendering_cell(certs: list[dict]) -> str:
    entries = [cert["identical_rendering"] for cert in certs]
    if not entries:
        return "—"
    if all(entry.get("skipped") for entry in entries):
        return "skipped"
    passed = sum(bool(entry["pass"]) for entry in entries)
    return f"{passed}/{len(entries)} exact"


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    root = Path(args.root)
    summary: dict = {"root": str(root), "seeds": list(SEEDS), "tasks": {}}
    header = ("| task | integrator | post-cue probe | cue probe | "
              "identical rendering | t4 memory demand | "
              + " | ".join(f"s{seed}" for seed in SEEDS) + " |")
    lines = [header,
             "|" + "---|" * (6 + len(SEEDS))]
    for task in TASKS:
        by_seed = {seed: _load(root, task, seed) for seed in SEEDS}
        certs = [cert for cert in by_seed.values() if cert is not None]
        summary["tasks"][task] = {
            "per_seed": {str(seed): cert for seed, cert in by_seed.items()},
            "n_found": len(certs),
        }
        verdicts = ["—" if by_seed[seed] is None else
                    ("PASS" if by_seed[seed]["overall_pass"] else "FAIL")
                    for seed in SEEDS]
        lines.append(
            f"| {task} | {_pooled(certs, 'integrator_probe')} | "
            f"{_pooled(certs, 'postcue_pixel_probe')} | "
            f"{_pooled(certs, 'cue_pixel_probe')} | "
            f"{_rendering_cell(certs)} | "
            f"{_pooled(certs, 'memory_demand')} | " + " | ".join(verdicts) + " |")

    markdown = "# V19 P1a certificate summary\n\n" + "\n".join(lines) + "\n"
    root.mkdir(parents=True, exist_ok=True)
    (root / "p1a_summary.json").write_text(json.dumps(summary, indent=2,
                                                      sort_keys=True))
    (root / "p1a_summary.md").write_text(markdown)
    print(markdown, flush=True)


if __name__ == "__main__":
    main()
