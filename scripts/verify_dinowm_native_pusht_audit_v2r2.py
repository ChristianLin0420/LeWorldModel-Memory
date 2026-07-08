#!/usr/bin/env python3
"""Independent read-only verification and compact aggregation for V2R2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/dinowm_native_pusht_audit_v2r2.yaml"
LOCK = CONFIG.with_suffix(".lock.json")
FORMAL = ROOT / "outputs/dinowm_native_pusht_audit_v2r2/formal"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    lock = json.loads(LOCK.read_text())
    assert digest(CONFIG) == lock["protocol_sha256"]
    for relative, expected in lock["code_sha256"].items():
        assert digest(ROOT / relative) == expected, relative
    summary = json.loads((FORMAL / "summary.json").read_text())
    provenance = json.loads((FORMAL / "provenance.json").read_text())
    selection = json.loads((FORMAL / "selection.json").read_text())
    health = json.loads((FORMAL / "rollout_health.json").read_text())
    assert summary["status"] == "complete"
    assert summary["bank_matched_to_lewm"] is False
    assert summary["host_is_dinowm_noprop"] is False
    assert summary["same_bank_v1_status"] == "failed_preserved"
    assert provenance["protocol_sha256"] == lock["protocol_sha256"]
    assert provenance["paper_modified"] is False
    assert provenance["carrier_injection"] is False
    assert provenance["bank_matched_to_lewm"] is False
    assert health == summary["rollout_health"] and health["admitted"] is True
    assert all(gate["pass"] for gate in health["gates"].values())
    assert set(summary["claim_ledger"].values()) <= {"tested", "not_applicable"}

    compact_tasks = {}
    for key, task_summary in summary["tasks"].items():
        admission = summary["admissions"][key]
        assert admission["admitted"] is True
        assert all(gate["pass"] for gate in admission["gates"].values())
        assert json.loads((FORMAL / "results" / f"{key}.json").read_text()) \
            == task_summary
        with np.load(FORMAL / "results" / f"{key}.npz") as arrays:
            assert arrays["labels"].shape == (1680,)
            for age in (1, 4, 8, 15):
                assert arrays[f"age_{age}_features"].shape == (1680, 8064)
                assert arrays[f"age_{age}_separation"].shape == (120,)
                assert np.isfinite(arrays[f"age_{age}_features"]).all()
                assert np.isfinite(arrays[f"age_{age}_separation"]).all()
            assert arrays["cue_separation"].shape == (120,)
            assert np.all(arrays["cue_separation"] > 0)
        selections = selection["tasks"][key]
        train = [row for row in selections if row["split"] == "train"]
        validation = [row for row in selections
                      if row["split"] == "validation"]
        assert len(train) == 1200 and len(validation) == 480
        train_ids = {(row["source_split"], row["episode_index"])
                     for row in train}
        validation_ids = {(row["source_split"], row["episode_index"])
                          for row in validation}
        assert len(train_ids) == 1200 and len(validation_ids) == 480
        assert train_ids.isdisjoint(validation_ids)
        assert {row["source_split"] for row in selections} == {"train"}
        age_rows = {}
        for age in (1, 4, 8, 15):
            probe = task_summary["open_loop_decodability"][str(age)]
            separation = task_summary["paired_counterfactual_separation"][
                "ages"][str(age)]
            age_rows[str(age)] = {
                "balanced_accuracy": probe["balanced_accuracy"],
                "chance": probe["chance"],
                "accuracy_ci": probe[
                    "validation_accuracy_episode_bootstrap"],
                "transport_ratio": separation["transport_ratio"],
                "transport_ratio_ci": [
                    separation["transport_ratio_lower"],
                    separation["transport_ratio_upper"],
                ],
            }
        compact_tasks[key] = {
            "classes": task_summary["task"]["classes"],
            "cue_availability": admission["probes"]["cue"][
                "balanced_accuracy"],
            "teacher_endpoint_balanced_accuracy": admission["probes"][
                "predicted_endpoint"]["balanced_accuracy"],
            "teacher_endpoint_ceiling": admission["shortcut_ceiling"],
            "ages": age_rows,
        }

    v1_receipt = ROOT / (
        "outputs/dinowm_native_pusht_audit_v1/formal/stop_receipt.json")
    v2_receipt = ROOT / (
        "outputs/dinowm_native_pusht_audit_v2/formal/"
        "implementation_stop_receipt.json")
    v2r_receipt = ROOT / (
        "outputs/dinowm_native_pusht_audit_v2r/launch_stop_receipt.json")
    for receipt in (v1_receipt, v2_receipt, v2r_receipt):
        assert receipt.is_file()
    value = {
        "schema": "dinowm_native_distribution_verification_v2r2",
        "verified": True,
        "protocol_sha256": lock["protocol_sha256"],
        "status": summary["status"],
        "host": summary["host"],
        "bank_matched_to_lewm": False,
        "host_is_dinowm_noprop": False,
        "preserved_attempts": {
            "v1_stop_sha256": digest(v1_receipt),
            "v2_implementation_stop_sha256": digest(v2_receipt),
            "v2r_launch_stop_sha256": digest(v2r_receipt),
        },
        "rollout_health": {
            "episodes": health["episodes"],
            "one_step_model_to_copy_ratio": health[
                "model_to_copy_ratio"][0],
            "integrated_model_to_copy_ratio": health[
                "integrated_model_to_copy_ratio"],
            "integrated_true_action_advantage": health[
                "integrated_true_action_advantage"],
            "all_gates_pass": True,
        },
        "tasks": compact_tasks,
        "no_carrier_injection": True,
        "persistent_state_and_use": "not_applicable",
        "paper_modified": False,
        "artifact_sha256": {
            str(path.relative_to(FORMAL)): digest(path)
            for path in sorted(FORMAL.rglob("*"))
            if path.is_file() and path.name != "verification.json"
        },
    }
    destination = FORMAL / "verification.json"
    destination.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
