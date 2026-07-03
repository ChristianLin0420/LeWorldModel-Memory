#!/usr/bin/env python3
"""Run the V19 P1a construction-level certificates for one task and seed.

Generates the certificate banks, evaluates every clause, writes
``<output>/<task>/s<seed>/certificate.json``, and (unless --no-wandb) logs the
clause table, per-task figures, and annotated rollout videos to wandb.

A failing certificate is a *result*, not an error: the exit code is 0 either
way and the PASS/FAIL verdict is printed for the launcher to aggregate.
Setting V19_P1A_SMOKE=1 shrinks the banks for end-to-end smoke testing.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.tasks_v19 import TASKS, make_task
from lewm.tasks_v19.certify import run_certificates
from lewm.tasks_v19.wandb_utils import (log_certificates, log_rollout_video,
                                        log_task_figures)

FULL_SIZES = {"e_train": 512, "e_eval": 256}
SMOKE_SIZES = {"e_train": 96, "e_eval": 48}
VIZ_EPISODES = 20  # enough for the t4 trajectory fan; cheap for the others


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", default="outputs/v19_p1a")
    parser.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    parser.add_argument("--no-wandb", dest="wandb", action="store_false")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--project", default="lewm-v19")
    return parser.parse_args(argv)


def _flatten_certificate(cert: dict) -> dict:
    summary: dict[str, float | bool | None] = {}
    for stream, clauses in cert["streams"].items():
        for name, clause in clauses.items():
            summary[f"{stream}/{name}/value"] = clause["value"]
            if clause["pass"] is not None:
                summary[f"{stream}/{name}/pass"] = clause["pass"]
    rendering = cert["identical_rendering"]
    summary["identical_rendering/value"] = rendering["value"]
    summary["identical_rendering/pass"] = rendering["pass"]
    summary["overall_pass"] = cert["overall_pass"]
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    smoke = os.environ.get("V19_P1A_SMOKE") == "1"
    sizes = SMOKE_SIZES if smoke else FULL_SIZES
    out_dir = Path(args.output) / args.task / f"s{args.seed}"
    task = make_task(args.task)

    print(f"[v19-p1a] certifying task={args.task} seed={args.seed} "
          f"e_train={sizes['e_train']} e_eval={sizes['e_eval']} smoke={smoke}",
          flush=True)
    cert = run_certificates(task, args.seed, out_dir, **sizes)
    verdict = "PASS" if cert["overall_pass"] else "FAIL"
    print(f"[v19-p1a] certificate {verdict}: task={args.task} seed={args.seed} "
          f"-> {out_dir / 'certificate.json'}", flush=True)

    if args.wandb:
        import wandb  # reads WANDB_API_KEY from the environment; never print it

        run = wandb.init(
            project=args.project, entity=args.entity,
            name=f"p1a-{args.task}-s{args.seed}", group=f"p1a-{args.task}",
            tags=["p1a", "v19"],
            config={**task.describe(), "seed": args.seed, "smoke": smoke, **sizes})
        try:
            log_certificates(run, cert)
            for stream_index, stream in enumerate(("iid", "script")):
                viz = task.generate(stream, VIZ_EPISODES,
                                    args.seed * 1000 + 10 + stream_index)
                log_task_figures(run, task, viz, key=f"figures/{stream}")
                log_rollout_video(run, viz, n=3, key=f"rollouts/{stream}")
            run.summary.update(_flatten_certificate(cert))
        finally:
            run.finish()
    sys.exit(0)


if __name__ == "__main__":
    main()
