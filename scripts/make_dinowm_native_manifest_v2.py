#!/usr/bin/env python3
"""Create the deterministic extracted-subset manifest for DINO-WM V2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    required_scalars = {
        f"{split}/{name}"
        for split in ("train", "val")
        for name in ("states.pth", "rel_actions.pth", "velocities.pth",
                     "seq_lengths.pkl")
    }
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    relative = {str(path.relative_to(root)) for path in paths}
    missing = required_scalars - relative
    if missing:
        raise RuntimeError(f"required extracted files missing: {sorted(missing)}")
    if any(path.name in {"tokens.pth", "abs_actions.pth"} for path in paths):
        raise RuntimeError("manifest subset unexpectedly includes unused large files")
    records = [{
        "relative_path": str(path.relative_to(root)),
        "size": path.stat().st_size,
        "sha256": digest(path),
    } for path in paths]
    value = {
        "schema": "dinowm_pusht_extracted_manifest_v1",
        "scope": (
            "Required native inference subset only: states, relative actions, "
            "velocities, sequence lengths, and RGB MP4 observations"),
        "root": "pusht_noise",
        "file_count": len(records),
        "total_size": sum(record["size"] for record in records),
        "excluded_by_design": ["tokens.pth", "abs_actions.pth"],
        "files": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value[key] for key in (
        "schema", "file_count", "total_size")}, indent=2))


if __name__ == "__main__":
    main()
