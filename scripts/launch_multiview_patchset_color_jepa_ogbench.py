#!/usr/bin/env python3
"""Launch temporal-coverage salient patch-set JEPA cells on multiple GPUs."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import launch_masked_evidence_jepa_ogbench as launcher  # noqa: E402


launcher.RUNNER = ROOT / "scripts" / "run_multiview_patchset_color_jepa_ogbench.py"
launcher.DEFAULT_OUTPUT = ROOT / "outputs" / "multiview_patchset_color_jepa_weak_v1"


if __name__ == "__main__":
    launcher.main()
