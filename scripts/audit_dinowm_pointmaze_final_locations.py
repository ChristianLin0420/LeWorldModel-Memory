#!/usr/bin/env python3
"""Index final Wave 3 cell locations without mutating locked artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "outputs/dinowm_pointmaze_wave3/formal"
ARMS = ("none", "gru", "lstm", "ssm", "fixed_trust")
SEEDS = range(5)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    destination = FORMAL / "final_location_index.json"
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite {destination}")
    records: dict[str, Any] = {}
    stale_locator_count = 0
    for arm in ARMS:
        for seed in SEEDS:
            directory = FORMAL / "cells" / arm / f"s{seed}"
            manifest_path = directory / "manifest.json"
            if not manifest_path.is_file():
                raise RuntimeError(f"missing completed cell: {manifest_path}")
            manifest = json.loads(manifest_path.read_text())
            for name, expected in manifest["artifacts"].items():
                path = directory / name
                if not path.is_file():
                    raise RuntimeError(f"missing final artifact: {path}")
                actual_size = path.stat().st_size
                actual_sha256 = digest(path)
                if actual_size != expected["size"] \
                        or actual_sha256 != expected["sha256"]:
                    raise RuntimeError(f"artifact differs: {path}")
                recorded_path = expected["path"]
                final_path = str(path.relative_to(ROOT))
                stale_locator = recorded_path != final_path
                stale_locator_count += int(stale_locator)
                records[final_path] = {
                    "size": actual_size,
                    "sha256": actual_sha256,
                    "recorded_pre_rename_path": recorded_path,
                    "recorded_locator_is_pre_rename": stale_locator,
                }
    result = {
        "schema": "dinowm_pointmaze_wave3_final_location_index_v1",
        "status": "verified",
        "cell_count": len(ARMS) * len(tuple(SEEDS)),
        "artifact_count": len(records),
        "stale_pre_rename_locator_count": stale_locator_count,
        "audit_note": (
            "Cell manifests were written in staging directories before atomic "
            "rename, so their locator strings preserve the pre-rename path. "
            "Sizes and SHA-256 digests are correct. This supplemental index "
            "records and verifies the immutable final locations without "
            "modifying any cell manifest."
        ),
        "artifacts": records,
    }
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in (
        "schema", "status", "cell_count", "artifact_count",
        "stale_pre_rename_locator_count")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
