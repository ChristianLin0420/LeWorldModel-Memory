#!/usr/bin/env python3
"""Train one V21 X0b/X1 cell (docs/V21_PROPOSAL.md 4/X0.3, X1).

A thin wrapper over the frozen V20 W1 trainer (scripts/train_v20_w1.py) that
extends the carrier registry with the X0b baseline-parity arms:

  acgru_h64_l1 / ... / acgru_h160_l10   ac-GRU width x lr sweep cells
  acgru_chrono_l1|_l3|_l10              the symmetric-repair control
  gdelta_l1|_l3|_l10                    the gated delta-rule rival
  acssm                                 reinstated V19 recipe
  lkc_rfix / acgru / none               pass-through (X1 wave)

The ``_lN`` suffix is a LABEL (own run directory per sweep cell); the actual
learning rate comes from --lr, which the launcher sets from the same suffix.
Everything else — host, gates, telemetry, checkpoint, eval export — is the
W1 trainer verbatim on the vicreg host.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.v21_carriers import make_carrier_v21
import scripts.train_v20_w1 as w1

SWEEP_WIDTHS = (64, 102, 160)
LR_SUFFIXES = {"l1": 1e-4, "l3": 3e-4, "l10": 1e-3}

X0_ARMS = tuple(
    [f"acgru_h{width}_{suffix}" for width in SWEEP_WIDTHS
     for suffix in LR_SUFFIXES]
    + [f"acgru_chrono_{suffix}" for suffix in LR_SUFFIXES]
    + [f"gdelta_{suffix}" for suffix in LR_SUFFIXES]
    + ["acssm"])


def arm_lr(arm: str) -> float:
    """The learning rate a sweep label encodes (default: V19 recipe)."""
    for suffix, lr in LR_SUFFIXES.items():
        if arm.endswith(f"_{suffix}"):
            return lr
    return 3e-4


def main(argv: Iterable[str] | None = None) -> None:
    w1.ARMS = (*w1.ARMS, *X0_ARMS)
    w1.make_carrier = make_carrier_v21
    w1.main(argv)


if __name__ == "__main__":
    main()
