from __future__ import annotations

import numpy as np
import json
from pathlib import Path
import torch
import yaml
import pytest

from scripts import aggregate_paper_a_matched_token as aggregate
from scripts import launch_paper_a_matched_token as launcher
from scripts.paper_a_matched_token_spec import DEFAULT_SPEC, validate_spec
from scripts.train_paper_a_matched_token import _aligned_latent, _nuisance
from scripts import evaluate_paper_a_matched_token_use as use_eval
from lewm.models.frozen_swap_carriers import make_frozen_carrier
from lewm.official_tasks.artifacts import write_npz_with_sidecar
from scripts.train_frozen_official_swap import state_digest
from scripts.paper_a_matched_token_spec import sha256_file


def test_matched_token_protocol_validates_before_lock() -> None:
    spec = yaml.safe_load(DEFAULT_SPEC.read_text())
    validate_spec(spec, verify_inputs=False)
    assert spec["token"]["classes"] == [
        "vertical-bar", "horizontal-bar", "x", "plus"]
    assert spec["admission"]["cue_min_class_recall_min"] == 0.70
    assert spec["tworoom_use"]["target"] == "token"


def test_matched_token_grid_sizes_are_exact() -> None:
    carriers = launcher.carrier_commands(DEFAULT_SPEC, DEFAULT_SPEC)
    uses = launcher.use_commands(DEFAULT_SPEC, DEFAULT_SPEC)
    assert len(carriers) == 75
    assert len({(host, arm, seed) for host, arm, seed, _ in carriers}) == 75
    assert len(uses) == 25
    assert len({(arm, seed) for arm, seed, _ in uses}) == 25


def test_token_alignment_changes_only_cue_interval() -> None:
    base = {"z_base": np.zeros((2, 20, 192), dtype=np.float32),
            "episode_index": np.array([3, 5]), "local_start": np.array([1, 2])}
    cue = {"z_cue": np.ones((2, 3, 192), dtype=np.float32),
           "episode_index": base["episode_index"],
           "local_start": base["local_start"],
           "cue_on": np.array([8, 8]), "cue_off": np.array([11, 11])}
    value = _aligned_latent(base, cue)
    assert np.all(value[:, 8:11] == 1)
    assert np.all(value[:, :8] == 0) and np.all(value[:, 11:] == 0)


def test_token_nuisance_and_bootstrap_are_deterministic() -> None:
    location = np.repeat(np.arange(4), 4)
    result = _nuisance(np.arange(16) % 3 == 0, location)
    assert len(result["per_location_accuracy"]) == 4
    correct = np.zeros((3, 5, 5, 3, 480), dtype=np.float32)
    joint = np.tile(np.repeat(np.arange(16), 30), (3, 1))
    nuisance = joint % 4
    correct[..., :120] = 1
    first = aggregate._bootstrap(correct, joint, nuisance, 32, 9)
    second = aggregate._bootstrap(correct, joint, nuisance, 32, 9)
    for left, right in zip(first, second):
        assert np.array_equal(left, right)


def _use_spec() -> dict:
    return {
        "study": "paper-a-matched-token-v1", "_lock": {"token": "lock"},
        "tworoom_use": {
            "physics_seed": 0, "success_radius": 16.0,
            "oracle_success_min": .90, "oracle_per_class_success_min": .90,
            "off_diagonal_false_success_max": .05,
            "replay_fidelity_min": .99,
            "upstream_environment": {"revision": "a" * 40},
        },
    }


def _valid_use_deck(path: Path, spec: dict) -> tuple[Path, dict]:
    joint = np.repeat(np.arange(16, dtype=np.int64), 30)
    token, location = joint // 4, joint % 4
    distance = np.full((480, 4, 4), 100.0, dtype=np.float32)
    distance[:, np.arange(4), np.arange(4)] = 1.0
    success = (distance < 16.0).astype(np.int8)
    arrays = {
        "z": np.zeros((480, 20, 192), np.float32),
        "actions": np.zeros((480, 19, 10), np.float32),
        "token_label": token, "location_label": location,
        "combination_label": joint, "episode_index": np.arange(480, dtype=np.int64),
        "local_start": np.zeros(480, dtype=np.int64),
        "global_frame_indices": (np.arange(480)[:, None] * 100
                                 + np.arange(20)[None] * 5).astype(np.int64),
        "decision_position": np.zeros((480, 2), np.float32),
        "goal_waypoints": np.zeros((4, 2), np.float32),
        "success_matrix": success, "distance_matrix": distance,
        "controller_target_success": np.ones((480, 4), np.int8),
        "controller_final_state": np.zeros((480, 4, 2), np.float32),
        "reset_replay": np.ones((480, 4), np.int8),
        "random_choice": token.copy(),
    }
    record = write_npz_with_sidecar(path, arrays, {
        "schema": "paper_a_matched_token_tworoom_use_deck_v1",
        "study": spec["study"], "lock": spec["_lock"], "episodes": 480,
        "target": "token", "nuisance": "location", "cue_age": 15,
        "fresh_exclusion_count": 3360, "fresh_zero_overlap": True,
        "physical_gpu": 0,
        "fixed_physics_seed": 0,
    })
    gate = {
        "schema_version": 1, "study": spec["study"], "lock": spec["_lock"],
        "status": "admitted", "admitted": True, "episodes": 480,
        "oracle_success_min": .90, "oracle_per_class_success_min": .90,
        "off_diagonal_false_success_max": .05, "replay_fidelity_min": .99,
        "vendor_commit": "a" * 40, "vendor_clean": True,
        "frozen_host_unchanged": True, "frozen_host_sha256_before": "b" * 64,
        "frozen_host_sha256_after": "b" * 64,
        "oracle_executed_success": 1.0,
        "oracle_per_class_executed_success": [1.0] * 4,
        "realized_random_executed_success": 1.0,
        "off_diagonal_false_success": 0.0, "reset_replay_fidelity": 1.0,
        "controller_selected_target_success": 1.0, "deck": record,
        "raw_action_mean": [0.0, 0.0], "raw_action_std_ddof1": [1.0, 1.0],
        "raw_action_count": [100, 100],
    }
    return path.with_name("deck_gate.json"), gate


def test_token_use_deck_rejects_tampered_gate(tmp_path, monkeypatch) -> None:
    spec = _use_spec()
    deck = tmp_path / "deck.npz"
    gate_path, gate = _valid_use_deck(deck, spec)
    gate_path.write_text(json.dumps(gate))
    admission_path = tmp_path / "tworoom_manifest.json"
    admission_path.write_text(json.dumps({
        "lock": spec["_lock"], "status": "admitted",
        "provenance": {"raw_action_mean": [0.0, 0.0],
                       "raw_action_std_ddof1": [1.0, 1.0],
                       "raw_action_count": [100, 100]}}))
    monkeypatch.setattr(use_eval, "deck_path", lambda _: deck)
    monkeypatch.setattr(use_eval, "gate_path", lambda _: gate_path)
    monkeypatch.setattr(use_eval, "host_manifest_path",
                        lambda *_: admission_path)
    monkeypatch.setattr(use_eval, "_record", lambda path: {
        "path": str(path), "sha256": sha256_file(path)})
    arrays, _ = use_eval._deck(spec)
    assert arrays["success_matrix"].shape == (480, 4, 4)
    gate["oracle_per_class_executed_success"][2] = 0.0
    gate_path.write_text(json.dumps(gate))
    with pytest.raises(ValueError, match="thresholds"):
        use_eval._deck(spec)


def test_token_use_carrier_rejects_mismatched_embedded_metrics(
        tmp_path, monkeypatch) -> None:
    directory = tmp_path / "cell"
    directory.mkdir()
    spec = {"study": "paper-a-matched-token-v1", "_lock": {"x": 1}}
    carrier = make_frozen_carrier("none", 192, 10)
    metrics = {
        "schema_version": 1, "study": spec["study"], "lock": spec["_lock"],
        "host": "tworoom", "branch": "matched-composite-token-fixed-endpoint",
        "arm": "none", "seed": 0, "target": "token", "nuisance": "location",
        "frozen_host_unchanged": True,
        "validation_labels_used_for_fitting": False,
        "carrier_state_sha256": state_digest(carrier),
    }
    (directory / "metrics.json").write_text(json.dumps(metrics))
    (directory / "history.csv").write_text("epoch,loss,lr\n")
    torch.save({"carrier_state_dict": carrier.state_dict(),
                "metrics": {**metrics, "seed": 99}}, directory / "carrier.pt")
    artifacts = {name: {"path": filename, "sha256": sha256_file(directory / filename)}
                 for name, filename in (("metrics", "metrics.json"),
                                        ("checkpoint", "carrier.pt"),
                                        ("history", "history.csv"))}
    manifest = {"schema_version": 1, "study": spec["study"],
                "lock": spec["_lock"], "host": "tworoom", "arm": "none",
                "seed": 0, "artifacts": artifacts}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(use_eval, "carrier_directory",
                        lambda *_: directory)
    monkeypatch.setattr(use_eval, "_record", lambda path: {
        "path": path.name, "sha256": sha256_file(path)})
    with pytest.raises(ValueError, match="payload"):
        use_eval._carrier(spec, "none", 0, torch.device("cpu"))
