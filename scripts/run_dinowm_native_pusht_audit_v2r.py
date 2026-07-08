#!/usr/bin/env python3
"""Locked implementation revision for DINO-WM native-distribution V2.

The scientific executor is unchanged.  This wrapper verifies the V2R lock,
creates the two declared artifact subdirectories, and only then invokes it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.run_dinowm_native_pusht_audit_v1 import (
    ROOT,
    _load_locked_config,
    _resolve,
)
from scripts.run_dinowm_native_pusht_audit_v2 import execute


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/dinowm_native_pusht_audit_v2r.yaml")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("refusing metric-bearing execution without --execute")
    config_path = args.config.resolve()
    config, _ = _load_locked_config(config_path)
    artifact_root = _resolve(ROOT, config["artifacts"]["root"])
    (artifact_root / "admission").mkdir(parents=True, exist_ok=False)
    (artifact_root / "results").mkdir(parents=True, exist_ok=False)
    print(json.dumps(execute(config_path), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
