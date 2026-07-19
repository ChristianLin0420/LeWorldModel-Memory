#!/usr/bin/env python3
"""Launch random-target ME-JEPA cells using the standard multi-GPU launcher."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import launch_masked_evidence_jepa_ogbench as launcher  # noqa: E402


launcher.RUNNER = ROOT / "scripts" / "run_random_target_jepa_ogbench.py"
launcher.DEFAULT_OUTPUT = ROOT / "outputs" / "random_target_jepa_ogbench_v1"


if __name__ == "__main__":
    launcher.main()
