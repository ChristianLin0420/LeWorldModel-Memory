from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from scripts import audit_paper_a_cross_wave_completion as audit


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _minimal_output_configs(root: Path) -> None:
    configs = root / "configs"
    configs.mkdir(parents=True)
    for name, output in (
            ("dinowm_wave2_spatial_carrier_v1_1.yaml", "outputs/wave2"),
            ("dinowm_pointmaze_wave3.yaml", "outputs/wave3")):
        (configs / name).write_text(yaml.safe_dump({
            "artifacts": {"root": output, "formal": "formal"},
        }))


def test_sha256_sidecar_accepts_exact_identity_and_rejects_tamper(
        tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    target.write_text('{"status":"verified"}\n')
    sidecar = tmp_path / "receipt.sha256"
    sidecar.write_text(f"{_sha(target)}  {target.name}\n")
    assert audit.verify_sha256_sidecar(
        target, sidecar, audit.HashVerifier()) == _sha(target)

    target.write_text('{"status":"tampered","extra":true}\n')
    with pytest.raises(audit.AuditFailure, match="sidecar differs"):
        audit.verify_sha256_sidecar(target, sidecar, audit.HashVerifier())


def test_bootstrap_contract_is_fail_closed() -> None:
    record = {"draws": 20_000, "paired": True,
              "native_episode_clusters": 120}
    audit.verify_bootstrap_record(record, native_clusters=120)
    with pytest.raises(audit.AuditFailure, match="draw count"):
        audit.verify_bootstrap_record({**record, "draws": 19_999},
                                      native_clusters=120)
    with pytest.raises(audit.AuditFailure, match="episode unit"):
        audit.verify_bootstrap_record(record, native_clusters=121)


def test_receipt_requires_execute_is_atomic_and_refuses_overwrite(
        tmp_path: Path) -> None:
    _minimal_output_configs(tmp_path)
    destination = Path("outputs/cross/receipt.json")
    payload = {"schema": "fixture", "status": "complete"}
    assert audit.emit_receipt(
        tmp_path, destination, payload, execute=False) is False
    assert not (tmp_path / destination).exists()

    assert audit.emit_receipt(
        tmp_path, destination, payload, execute=True) is True
    assert json.loads((tmp_path / destination).read_text()) == payload
    with pytest.raises(audit.AuditFailure, match="already exists"):
        audit.emit_receipt(tmp_path, destination, payload, execute=True)


def test_receipt_cannot_enter_experiment_root(tmp_path: Path) -> None:
    _minimal_output_configs(tmp_path)
    payload = {"schema": "fixture", "status": "complete"}
    with pytest.raises(audit.AuditFailure, match="experiment root"):
        audit.emit_receipt(
            tmp_path, Path("outputs/wave2/receipt.json"), payload,
            execute=True)


def test_repository_path_rejects_escape(tmp_path: Path) -> None:
    with pytest.raises(audit.AuditFailure, match="leaves repository"):
        audit.repository_path(tmp_path, "../outside")


def test_empty_repository_fails_before_any_write(tmp_path: Path) -> None:
    with pytest.raises(audit.AuditFailure, match="missing Wave 1 receipt"):
        audit.audit_repository(tmp_path)
    assert not list(tmp_path.rglob("receipt.json"))


def test_missing_official_verification_is_rejected(tmp_path: Path) -> None:
    formal = tmp_path / "formal"
    formal.mkdir()
    with pytest.raises(audit.AuditFailure, match="missing official verification"):
        audit._require_official_verification(
            formal, audit.HashVerifier(), "expected", "p" * 64,
            {"cells/x": "a" * 64}, {"summary_sha256": "s" * 64})


def test_valid_official_verification_is_accepted_read_only(
        tmp_path: Path) -> None:
    formal = tmp_path / "formal"
    formal.mkdir()
    value = {
        "schema": "expected",
        "verified": True,
        "protocol_sha256": "p" * 64,
        "artifact_sha256": {"cells/x": "a" * 64},
        "summary_sha256": "s" * 64,
    }
    path = formal / "verification.json"
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    before = path.read_bytes()
    digest = audit._require_official_verification(
        formal, audit.HashVerifier(), "expected", "p" * 64,
        {"cells/x": "a" * 64}, {"summary_sha256": "s" * 64})
    assert digest == _sha(path)
    assert path.read_bytes() == before
