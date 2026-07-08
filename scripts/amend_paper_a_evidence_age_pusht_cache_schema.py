#!/usr/bin/env python3
"""Normalize the strict PushT cache schema before any carrier metric exists.

The locked preparer records ``local_start`` in each PushT age cache after using
it to verify exact episode/local-start identity.  The locked trainer accepts
only the six numerical arrays it consumes and therefore rejects that redundant
identity array.  This amendment preserves the original artifacts byte-for-byte
in an archive, removes only ``local_start`` from the trainer-facing copies,
updates their hash records, and writes a complete before/after receipt.

No model, label, latent, action, cue boundary, or episode index is changed.
Execution is refused if any strict PushT carrier metric already exists.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.official_tasks.artifacts import (  # noqa: E402
    atomic_text,
    load_verified_npz,
    sha256_array,
    sha256_arrays,
    sha256_file,
    stable_json,
    write_npz_with_sidecar,
)
from scripts.paper_a_evidence_age_spec import (  # noqa: E402
    DEFAULT_SHA,
    DEFAULT_SPEC,
    PUSHT_TASKS,
    load_locked_spec,
    output_root,
)
from scripts.prepare_paper_a_evidence_age_strict import (  # noqa: E402
    strict_cache_path,
)


AMENDMENT_ID = "pusht-cache-schema-local-start-v1"
REQUIRED = {
    "z", "actions", "labels", "cue_on", "cue_off", "episode_index",
}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--sha", type=Path, default=DEFAULT_SHA)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def _artifact_index(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for record in manifest.get("artifacts", []):
        path = Path(record["path"]).resolve()
        if path in result:
            raise ValueError(f"duplicate artifact record: {path}")
        result[path] = record
    return result


def _verify_no_pusht_metrics(strict_root: Path) -> None:
    carrier_root = strict_root / "carriers" / "pusht"
    metrics = list(carrier_root.rglob("metrics.json")) if carrier_root.exists() else []
    manifests = list(carrier_root.rglob("manifest.json")) if carrier_root.exists() else []
    other_artifacts = ([path for path in carrier_root.rglob("*") if path.is_file()]
                       if carrier_root.exists() else [])
    if metrics or manifests or other_artifacts:
        raise RuntimeError(
            "schema amendment must precede every PushT carrier metric: "
            f"metrics={len(metrics)}, manifests={len(manifests)}, "
            f"other_files={len(other_artifacts)}")


def _preflight(spec: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    strict_root = output_root(spec, "strict")
    _verify_no_pusht_metrics(strict_root)
    ages = [int(value) for value in spec["strict_fixed_endpoint"]["pusht"]["ages"]]
    candidates: list[dict[str, Any]] = []
    manifests: dict[str, Any] = {}
    for task in PUSHT_TASKS:
        manifest_path = strict_root / "cache" / "pusht" / task / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("lock") != spec["_lock"] \
                or manifest.get("status") != "admitted" \
                or manifest.get("all_ages_admitted") is not True:
            raise RuntimeError(f"PushT cache is not formally admitted: {manifest_path}")
        if manifest.get("cache_schema_amendment") is not None:
            raise RuntimeError(f"cache already amended: {manifest_path}")
        artifact_index = _artifact_index(manifest)
        manifests[task] = {
            "path": manifest_path,
            "value": manifest,
            "sha256": sha256_file(manifest_path),
            "admission_before": json.loads(json.dumps(manifest["admission"])),
        }
        replay = manifest.get("replay", {})
        for split in ("train", "validation"):
            expected_local_hash = replay[split]["local_start_sha256"]
            expected_episode_hash = replay[split]["episode_index_sha256"]
            reference_local_hash = None
            reference_identity = None
            for age in ages:
                path = strict_cache_path(spec, "pusht", task, split, age)
                arrays, sidecar = load_verified_npz(path)
                if set(arrays) != REQUIRED | {"local_start"}:
                    raise ValueError(
                        f"unexpected pre-amendment arrays {path}: {sorted(arrays)}")
                if sidecar.get("lock") != spec["_lock"] \
                        or sidecar.get("host") != "pusht" \
                        or sidecar.get("task") != task \
                        or sidecar.get("split") != split \
                        or sidecar.get("age") != age:
                    raise ValueError(f"cache identity mismatch: {path}")
                local_hash = sha256_array(arrays["local_start"])
                episode_hash = sha256_array(arrays["episode_index"])
                if local_hash != expected_local_hash \
                        or episode_hash != expected_episode_hash:
                    raise ValueError(f"paired identity hash mismatch: {path}")
                if reference_local_hash is not None \
                        and local_hash != reference_local_hash:
                    raise ValueError(
                        f"local_start changes across registered ages: {task}/{split}")
                reference_local_hash = local_hash
                identity = {
                    name: sha256_array(arrays[name])
                    for name in ("actions", "labels", "episode_index",
                                 "local_start")
                }
                if reference_identity is not None and identity != reference_identity:
                    raise ValueError(
                        f"paired identity changes across ages: {task}/{split}")
                reference_identity = identity
                cue_on = np.asarray(arrays["cue_on"], dtype=np.int64)
                cue_off = np.asarray(arrays["cue_off"], dtype=np.int64)
                if not np.all(cue_off - cue_on == 3) \
                        or not np.all(19 - cue_off == age):
                    raise ValueError(f"cue placement mismatch: {path}")
                record = artifact_index.get(path.resolve())
                sidecar_path = path.with_suffix(path.suffix + ".json")
                if record is None \
                        or record.get("sha256") != sha256_file(path) \
                        or record.get("sidecar_sha256") != sha256_file(sidecar_path):
                    raise ValueError(f"manifest artifact hash mismatch: {path}")
                consumed = {name: arrays[name] for name in sorted(REQUIRED)}
                candidates.append({
                    "task": task,
                    "split": split,
                    "age": age,
                    "path": path,
                    "sidecar_path": sidecar_path,
                    "metadata": {
                        key: value for key, value in sidecar.items()
                        if key not in ("arrays", "artifact")
                    },
                    "old_record": dict(record),
                    "consumed_sha256": sha256_arrays(consumed),
                    "local_start_sha256": local_hash,
                })
    expected = len(PUSHT_TASKS) * 2 * len(ages)
    if len(candidates) != expected:
        raise RuntimeError(f"expected {expected} caches, found {len(candidates)}")
    return candidates, manifests


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    spec = load_locked_spec(args.spec, args.sha)
    candidates, manifests = _preflight(spec)
    strict_root = output_root(spec, "strict")
    amendment_root = strict_root / "amendments" / AMENDMENT_ID
    receipt_path = amendment_root / "receipt.json"
    if amendment_root.exists():
        raise FileExistsError(f"amendment path already exists: {amendment_root}")
    if not args.execute:
        print(
            f"[evidence-age/amendment] preflight passed for {len(candidates)} "
            "PushT caches; rerun with --execute",
            flush=True)
        return

    original_root = amendment_root / "original"
    for task, item in manifests.items():
        destination = original_root / task / "manifest.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item["path"], destination)
    for item in candidates:
        destination = (original_root / item["task"] / item["split"] /
                       f"age-{item['age']}.npz")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item["path"], destination)
        shutil.copy2(item["sidecar_path"],
                     destination.with_suffix(destination.suffix + ".json"))

    new_records: dict[Path, dict[str, Any]] = {}
    receipt_files = []
    for item in candidates:
        arrays, _ = load_verified_npz(item["path"])
        consumed = {name: arrays[name] for name in sorted(REQUIRED)}
        if sha256_arrays(consumed) != item["consumed_sha256"]:
            raise RuntimeError(f"consumed arrays changed during amendment: {item['path']}")
        new_record = write_npz_with_sidecar(
            item["path"], consumed, item["metadata"],
            compression_level=1, overwrite=True)
        after, _ = load_verified_npz(item["path"])
        if set(after) != REQUIRED \
                or sha256_arrays(after) != item["consumed_sha256"]:
            raise RuntimeError(f"post-amendment verification failed: {item['path']}")
        new_records[item["path"].resolve()] = new_record
        receipt_files.append({
            "path": _relative(item["path"]),
            "task": item["task"], "split": item["split"],
            "age": item["age"],
            "removed_array": "local_start",
            "removed_array_sha256": item["local_start_sha256"],
            "consumed_arrays_sha256_before_after": item["consumed_sha256"],
            "old_artifact_sha256": item["old_record"]["sha256"],
            "new_artifact_sha256": new_record["sha256"],
            "old_sidecar_sha256": item["old_record"]["sidecar_sha256"],
            "new_sidecar_sha256": new_record["sidecar_sha256"],
        })

    receipt_relative = _relative(receipt_path)
    manifest_receipts = []
    for task, item in manifests.items():
        manifest = item["value"]
        replaced = []
        for old_record in manifest["artifacts"]:
            path = Path(old_record["path"]).resolve()
            if path not in new_records:
                raise RuntimeError(f"missing amended artifact record: {path}")
            replaced.append(new_records[path])
        manifest["artifacts"] = replaced
        manifest["cache_schema_amendment"] = {
            "id": AMENDMENT_ID,
            "reason": "remove redundant local_start rejected by locked trainer",
            "receipt": receipt_relative,
            "applied_before_any_pusht_carrier_metric": True,
            "consumed_numerical_arrays_unchanged": True,
            "locked_preparer_and_trainer_unchanged": True,
        }
        atomic_text(item["path"], stable_json(manifest), overwrite=True)
        reread = json.loads(item["path"].read_text())
        if reread.get("admission") != item["admission_before"]:
            raise RuntimeError(f"admission record changed: {item['path']}")
        manifest_receipts.append({
            "task": task,
            "path": _relative(item["path"]),
            "old_sha256": item["sha256"],
            "new_sha256": sha256_file(item["path"]),
            "admission_unchanged": True,
        })

    _verify_no_pusht_metrics(strict_root)
    failed_logs = []
    log_root = output_root(spec, "logs") / "strict-crossgpu-carriers"
    for path in sorted(log_root.glob("strict-carrier-pusht-*.log")):
        text = path.read_text(errors="replace")
        if "strict cache mismatch" not in text or "[evidence-age/strict] wrote" in text:
            raise RuntimeError(f"unexpected pre-amendment PushT log: {path}")
        failed_logs.append({
            "path": _relative(path), "sha256": sha256_file(path),
            "failure_stage": "locked cache schema validation before training",
        })
    if not failed_logs:
        raise RuntimeError("expected preserved PushT loader-failure logs")

    receipt = {
        "schema_version": 1,
        "study": spec["study"],
        "amendment_id": AMENDMENT_ID,
        "status": "complete-before-any-pusht-carrier-metric",
        "reason": (
            "The locked preparer retained local_start after exact paired-selection "
            "verification, while the locked trainer required only consumed arrays."
        ),
        "scope": "cache schema only; no numerical training/evaluation array changed",
        "lock": spec["_lock"],
        "amendment_script": {
            "path": _relative(Path(__file__)),
            "sha256": sha256_file(Path(__file__)),
        },
        "failed_attempts_before_amendment": failed_logs,
        "original_archive": _relative(original_root),
        "manifests": manifest_receipts,
        "files": receipt_files,
        "checks": {
            "all_original_artifacts_archived": True,
            "local_start_matches_admission_replay_hash": True,
            "episode_index_matches_admission_replay_hash": True,
            "actions_labels_episode_and_local_start_identical_across_ages": True,
            "cue_boundaries_match_registered_ages": True,
            "consumed_arrays_byte_identical_before_after": True,
            "admission_records_unchanged": True,
            "locked_preparer_unchanged": True,
            "locked_trainer_unchanged": True,
            "zero_pusht_carrier_metrics_before_after": True,
        },
    }
    atomic_text(receipt_path, stable_json(receipt))
    print(
        f"[evidence-age/amendment] normalized {len(candidates)} caches; "
        f"receipt {receipt_path}", flush=True)


if __name__ == "__main__":
    main()
