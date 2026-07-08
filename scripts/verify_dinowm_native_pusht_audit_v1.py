#!/usr/bin/env python3
"""Read-only verification/aggregation for the native DINO-WM audit artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/dinowm_native_pusht_audit_v1.yaml"
LOCK = CONFIG.with_suffix(".lock.json")
FORMAL = ROOT / "outputs/dinowm_native_pusht_audit_v1/formal"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    lock = json.loads(LOCK.read_text())
    assert sha256(CONFIG) == lock["protocol_sha256"]
    for relative, expected in lock["code_sha256"].items():
        assert sha256(ROOT / relative) == expected, relative
    data = json.loads((FORMAL / "data_contract.json").read_text())
    stop = json.loads((FORMAL / "stop_receipt.json").read_text())
    provenance = json.loads((FORMAL / "provenance.json").read_text())
    assert stop["status"] == "stopped_fail_closed"
    assert stop["no_post_hoc_adaptation"] is True
    assert not data["admitted"]
    assert data["action_distribution"]["admitted"] is True
    assert data["proprio_distribution"]["admitted"] is False
    assert provenance["protocol_sha256"] == lock["protocol_sha256"]
    assert provenance["paper_modified"] is False
    assert provenance["carrier_injection"] is False
    forbidden = [
        FORMAL / "teacher_features.npz",
        FORMAL / "rollout_health.json",
        FORMAL / "summary.json",
        FORMAL / "results",
    ]
    assert not any(path.exists() for path in forbidden)
    compact = {
        "schema": "dinowm_native_pusht_audit_verification_v1",
        "verified": True,
        "protocol_sha256": lock["protocol_sha256"],
        "status": stop["status"],
        "stop_reason": stop["reason"],
        "requested_dinowm_noprop_http_status": int(
            provenance["identities"]["missing_release_status"]["content"][0]),
        "fallback": provenance["fallback_identity"],
        "fallback_is_not_dinowm_noprop":
            provenance["fallback_is_not_dinowm_noprop"],
        "action_gate": data["action_distribution"]["gates"],
        "proprio_gate": data["proprio_distribution"]["gates"],
        "proprio_mean": data["proprio_distribution"]["mean"],
        "proprio_reference_mean":
            data["proprio_distribution"]["reference_mean"],
        "no_downstream_artifacts_after_stop": True,
        "artifact_sha256": {
            path.name: sha256(path)
            for path in sorted(FORMAL.glob("*.json"))
            if path.name != "verification.json"
        },
    }
    destination = FORMAL / "verification.json"
    destination.write_text(json.dumps(compact, indent=2, sort_keys=True) + "\n")
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
