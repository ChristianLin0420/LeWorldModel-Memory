#!/usr/bin/env python3
"""Second locked launch revision for native-distribution V2.

Scientific execution remains in the unchanged V2 executor.  This wrapper adds
the repository root before imports and creates only declared artifact folders.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_dinowm_native_pusht_audit_v1 import (  # noqa: E402
    _load_locked_config,
    _resolve,
)
from scripts.run_dinowm_native_pusht_audit_v2 import execute  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/dinowm_native_pusht_audit_v2r2.yaml")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        config_path = args.config.resolve()
        config = yaml.safe_load(config_path.read_text())
        required_paths = [
            config_path,
            _resolve(ROOT, config["dataset"]["archive_path"]),
            _resolve(ROOT, config["dataset"]["root"]),
            _resolve(ROOT, config["dataset"]["manifest_path"]),
            _resolve(ROOT, config["checkpoint"]["weights_path"]),
            _resolve(ROOT, config["checkpoint"]["config_path"]),
            _resolve(ROOT, config["source"]["repo_path"]),
            _resolve(ROOT, config["dino_encoder"]["repo_path"]),
            _resolve(ROOT, config["dino_encoder"]["weights_path"]),
            _resolve(ROOT, config["execution"]["decord_dependency"]),
            _resolve(ROOT, config["execution"]["dependency_manifest_path"]),
        ]
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            raise SystemExit(f"smoke-test missing paths: {missing}")
        smoke_root = ROOT / "outputs/dinowm_native_pusht_audit_v2r2/smoke"
        (smoke_root / "admission").mkdir(parents=True, exist_ok=False)
        (smoke_root / "results").mkdir(parents=True, exist_ok=False)
        if any(any((smoke_root / name).iterdir())
               for name in ("admission", "results")):
            raise SystemExit("smoke-test output directories are not empty")
        print(json.dumps({
            "status": "smoke_passed_before_lock",
            "config": str(config_path),
            "required_paths": len(required_paths),
            "empty_directories": [
                str(smoke_root / "admission"),
                str(smoke_root / "results"),
            ],
            "model_or_data_evaluation": False,
        }, indent=2, sort_keys=True))
        return
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
