#!/usr/bin/env python3
"""V21 F3 — the retention horizon as a design knob (§12/F3 registration).

On frozen lkc_rfix checkpoints, re-derive the fixed spectrum for each
evaluated episode length: every finite half-life is scaled by r = L / 64
(training length), which is exactly ``a_fixed ** (1/r)`` since
a = exp(-1/tau); the eigenvalue-1 hold channel is untouched (1^x = 1) and
every learned weight stays frozen.  Evaluated on the §11 delay banks with
the registered probe.  Confirmed-if (frozen in §12 before this ran):
rescaled >= frozen + 0.05 mean at L=128 (3 seeds), no loss (> -0.02) at
L=64.  Writes outputs/v21_x3/tau_rescale.{json,md}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.eval_v20_w2 as w2
from scripts.x3_delay_v21 import (LENGTHS, X3, build_delay_bank, probe_bank)

TRAIN_LENGTH = 64


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/v20_w3")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available()
        else "cpu")
    seeds = [int(value) for value in args.seeds.split(",")]

    frozen_ref = json.loads(
        (X3 / "delay_scaling.json").read_text())["curves"]

    curves: dict[str, Any] = {}
    for length in LENGTHS:
        print(f"[v21-f3] building L={length} bank", flush=True)
        bank = build_delay_bank(length)
        ratio = length / TRAIN_LENGTH
        scores = []
        for seed in seeds:
            model, carrier, checkpoint = w2.load_checkpoint(
                Path(args.root), "t1", "lkc_rfix", seed, device)
            with torch.no_grad():
                carrier.a_fixed.copy_(carrier.a_fixed.pow(1.0 / ratio))
            scores.append(probe_bank(model, carrier, checkpoint["host"],
                                     bank, device))
            del model, carrier
            if device.type == "cuda":
                torch.cuda.empty_cache()
        frozen = frozen_ref[f"lkc_rfix@L{length}"]["mean"]
        curves[f"L{length}"] = {
            "ratio": ratio,
            "rescaled_mean": float(np.mean(scores)),
            "rescaled_sd": float(np.std(scores)),
            "rescaled_scores": [round(s, 4) for s in scores],
            "frozen_mean": frozen,
            "delta": float(np.mean(scores) - frozen),
        }
        print(f"[v21-f3] L={length}: rescaled "
              f"{curves[f'L{length}']['rescaled_mean']:.3f} vs frozen "
              f"{frozen:.3f} (delta {curves[f'L{length}']['delta']:+.3f})",
              flush=True)

    confirmed = (curves["L128"]["delta"] >= 0.05
                 and curves["L64"]["delta"] > -0.02)
    report = {
        "schema_version": 1,
        "study": "v21-f3-tau-rescale",
        "intervention": "a_fixed ** (64/L) per evaluated length; hold "
                        "channel and all learned weights untouched",
        "registered_bar": "L128 delta >= +0.05 and L64 delta > -0.02",
        "CONFIRMED": bool(confirmed),
        "curves": curves,
    }
    (X3 / "tau_rescale.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = ["# V21 F3 — spectrum rescaled to the evaluated horizon", "",
             "| L | frozen | rescaled | delta |", "|---|---|---|---|"]
    for length in LENGTHS:
        row = curves[f"L{length}"]
        lines.append(f"| {length} | {row['frozen_mean']:.3f} | "
                     f"{row['rescaled_mean']:.3f} | {row['delta']:+.3f} |")
    lines.append("")
    lines.append(f"**Registered gate: {'CONFIRMED' if confirmed else 'NOT MET'}**")
    (X3 / "tau_rescale.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
