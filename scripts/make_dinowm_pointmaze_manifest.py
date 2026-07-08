#!/usr/bin/env python3
"""Create an immutable SHA-256 inventory of the official PointMaze archive."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "outputs/dinowm_pointmaze_wave3/downloads/point_maze.zip"
DATA = Path("/dev/shm/dinowm_pointmaze_wave3/point_maze")
OUTPUT = Path("/dev/shm/dinowm_pointmaze_wave3/extracted_manifest.json")
EXPECTED_ARCHIVE_SIZE = 718_363_945
EXPECTED_ARCHIVE_SHA256 = (
    "6c48ccf22c90b9af8dcf0e2cd70849aec8dd8e214ac5f1f09552bf8bc9494acc")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    if ARCHIVE.stat().st_size != EXPECTED_ARCHIVE_SIZE \
            or digest(ARCHIVE) != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError("official PointMaze archive identity failed")
    paths = sorted(path for path in DATA.rglob("*") if path.is_file())
    if len(paths) != 2003:
        raise RuntimeError(f"expected 2,003 extracted files, found {len(paths)}")
    records = []
    for index, path in enumerate(paths, 1):
        records.append({
            "path": str(path.relative_to(DATA)),
            "size": path.stat().st_size,
            "sha256": digest(path),
        })
        if index % 100 == 0:
            print(f"[pointmaze-manifest] {index}/{len(paths)}", flush=True)
    payload = {
        "schema": "official_dinowm_pointmaze_extraction_v1",
        "archive": {"path": str(ARCHIVE.relative_to(ROOT)),
                    "size": EXPECTED_ARCHIVE_SIZE,
                    "sha256": EXPECTED_ARCHIVE_SHA256},
        "root": str(DATA),
        "file_count": len(records),
        "total_bytes": sum(record["size"] for record in records),
        "files": records,
    }
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, OUTPUT)
    print(json.dumps({"manifest": str(OUTPUT),
                      "size": OUTPUT.stat().st_size,
                      "sha256": digest(OUTPUT),
                      "files": len(records),
                      "bytes": payload["total_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
