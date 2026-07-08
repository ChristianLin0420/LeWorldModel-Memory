#!/usr/bin/env python3
"""Seal the PushT cache-schema amendment and its failure history."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.official_tasks.artifacts import (  # noqa: E402
    atomic_text,
    sha256_file,
    stable_json,
)


AMENDMENT_ROOT = (
    ROOT / "outputs/paper_a_evidence_age_v1/strict/amendments"
    / "pusht-cache-schema-local-start-v1"
)


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def _copy_verified(source: Path, destination: Path,
                   expected: str) -> dict[str, str]:
    if not source.is_file() or sha256_file(source) != expected:
        raise ValueError(f"source hash mismatch: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copy2(source, destination)
    if sha256_file(destination) != expected:
        raise RuntimeError(f"archive copy mismatch: {destination}")
    return {"source": _relative(source), "archive": _relative(destination),
            "sha256": expected}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    receipt_path = AMENDMENT_ROOT / "receipt.json"
    seal_path = AMENDMENT_ROOT / "seal.json"
    sidecar_path = AMENDMENT_ROOT / "receipt.sha256"
    if seal_path.exists() or sidecar_path.exists():
        raise FileExistsError("amendment is already sealed")
    receipt = json.loads(receipt_path.read_text())
    receipt_hash = sha256_file(receipt_path)
    if receipt_hash != "07c6581c42e89c7b70765b262580237581fb6cac29c1bbaff73e3a90b9f79d03":
        raise ValueError("unexpected amendment receipt identity")
    failures = receipt.get("failed_attempts_before_amendment", [])
    if len(failures) != 17:
        raise ValueError(f"expected 17 failure logs, found {len(failures)}")
    if not args.execute:
        print("[evidence-age/seal] preflight passed; rerun with --execute")
        return

    archived_logs = []
    for record in failures:
        source = ROOT / record["path"]
        destination = AMENDMENT_ROOT / "failed-logs" / source.name
        archived_logs.append(_copy_verified(
            source, destination, record["sha256"]))

    protocol_files = []
    protocol_records = (
        receipt["amendment_script"],
        {"path": receipt["lock"]["path"],
         "sha256": receipt["lock"]["sha256"]},
        {"path": receipt["lock"]["sidecar"],
         "sha256": receipt["lock"]["sidecar_sha256"]},
        {"path": receipt["lock"]["implementation"]["path"],
         "sha256": receipt["lock"]["implementation"]["sha256"]},
    )
    for record in protocol_records:
        source = ROOT / record["path"]
        destination = AMENDMENT_ROOT / "protocol" / source.name
        protocol_files.append(_copy_verified(
            source, destination, record["sha256"]))

    atomic_text(
        sidecar_path,
        f"{receipt_hash}  receipt.json\n")
    seal = {
        "schema_version": 1,
        "study": receipt["study"],
        "amendment_id": receipt["amendment_id"],
        "status": "sealed",
        "receipt": {
            "path": _relative(receipt_path), "sha256": receipt_hash,
            "sidecar": _relative(sidecar_path),
            "sidecar_sha256": sha256_file(sidecar_path),
        },
        "failed_logs": archived_logs,
        "protocol_files": protocol_files,
        "original_cache_archive": receipt["original_archive"],
        "seal_script": {
            "path": _relative(Path(__file__)),
            "sha256": sha256_file(Path(__file__)),
        },
    }
    atomic_text(seal_path, stable_json(seal))
    print(f"[evidence-age/seal] wrote {seal_path}")


if __name__ == "__main__":
    main()
