#!/usr/bin/env python3
"""Independent Wave 2 v1.1 verification, including amendment hard links."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.verify_dinowm_wave2_spatial_carrier as base


CONFIG = ROOT / "configs/dinowm_wave2_spatial_carrier_v1_1.yaml"
LOCK = CONFIG.with_suffix(".lock.json")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    base.CONFIG = CONFIG
    base.LOCK = LOCK
    base.main()

    cfg = yaml.safe_load(CONFIG.read_text())
    formal = ROOT / cfg["artifacts"]["root"] / cfg["artifacts"]["formal"]
    cache = ROOT / cfg["artifacts"]["root"] / "cache"
    manifest = json.loads((cache / "manifest.json").read_text())
    amendment = manifest["amendment"]
    require(amendment["exact_original_layout_gate"]["pass"] is True
            and amendment["exact_original_layout_gate"]["value"] == 0.0,
            "exact original-layout replay gate differs")
    require(amendment["reusable_cache_layout_gate"]["pass"] is True
            and amendment["reusable_cache_layout_gate"]["value"]
            <= amendment["reusable_cache_layout_gate"]["threshold"]
            == 0.0001,
            "cache-layout numerical gate differs")
    require(manifest["carrier_outcomes_computed_before_gate"] is False,
            "amendment gate was not pre-outcome")
    for name, record in manifest["artifacts"].items():
        source = ROOT / record["parent_path"]
        destination = ROOT / record["path"]
        require(source.is_file() and destination.is_file()
                and os.path.samefile(source, destination),
                f"v1.1 cache is not a hard link: {name}")
        require(base.digest(source) == base.digest(destination)
                == record["sha256"], f"hard-link hash differs: {name}")

    verification_path = formal / "verification.json"
    verification = json.loads(verification_path.read_text())
    verification.update({
        "schema": "dinowm_wave2_spatial_verification_v1_1",
        "preoutcome_numerical_amendment_verified": True,
        "exact_original_layout_replay_max_abs": 0.0,
        "reusable_cache_layout_max_abs":
            amendment["reusable_cache_layout_gate"]["value"],
        "reusable_cache_layout_threshold": 0.0001,
        "parent_v1_failure_preserved": True,
        "cache_reused_by_hard_link": True,
        "carrier_outcomes_seen_before_amendment": False,
        "cache_manifest_sha256": base.digest(cache / "manifest.json"),
    })
    verification_path.write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n")
    print(json.dumps(verification, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
