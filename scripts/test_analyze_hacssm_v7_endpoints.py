#!/usr/bin/env python3
"""Focused tests for deterministic HACSSM-v7 endpoint replay."""

from __future__ import annotations

import json
import inspect
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.memory import HierarchicalActionConditionedMemory
from lewm.models.memory_model import MemoryLeWorldModel
import scripts.analyze_hacssm_v7_endpoints as analysis


def make_model() -> MemoryLeWorldModel:
    torch.manual_seed(17)
    model = MemoryLeWorldModel(
        img_size=8, patch_size=4, embed_dim=8, action_dim=3,
        encoder_layers=1, encoder_heads=2, predictor_layers=1, predictor_heads=2,
        predictor_norm="none", history_len=3, dropout=0.0, sigreg_projections=8,
        encoder_type="precomputed", memory_impl="hacssmv7", memory_mode="both",
        hier_loss_weight=0.02,
    ).cpu().float().eval()
    with torch.no_grad():
        memory = model.mem_hacssmv7
        memory.W_a.weight.normal_(std=0.15)
        memory.w_z.normal_(std=0.2)
        memory.w_e.normal_(std=0.2)
        memory.gate_bias.copy_(torch.tensor((0.4, -0.2)))
        memory.shrink_logits.copy_(torch.tensor((-0.3, 0.7)))
        memory.route_logits.copy_(torch.tensor((0.2, -0.4)))
        memory.W_o.weight.normal_(std=0.1)
    return model


def synthetic_batch():
    torch.manual_seed(23)
    observed = torch.randn(4, 32, 8, dtype=torch.float32)
    actions = torch.randn(4, 31, 3, dtype=torch.float32)
    return observed, actions


def test_replay_matches_native_and_exact_endpoints():
    model = make_model()
    observed, actions = synthetic_batch()
    memory = model.mem_hacssmv7
    with torch.inference_mode():
        z = model.encode(observed)
        native, native_details = model._inject(
            z, actions=actions, return_memory_details=True)
        learned, learned_details = analysis.memory_with_rho(
            memory, z, actions, memory.shrinkage())
        assert analysis.native_difference(
            native, native_details, learned, learned_details) == 0.0

        static, static_details = analysis.memory_with_rho(memory, z, actions, (0.0, 0.0))
        static_gate = torch.sigmoid(memory.gate_bias).view(1, 1, 2, 1)
        assert torch.equal(static_details["gates"], static_gate.expand_as(static_details["gates"]))
        assert not torch.equal(static, learned)

        dynamic, dynamic_details = analysis.memory_with_rho(memory, z, actions, (1.0, 1.0))
        old_mode = memory.v7_mode
        memory.v7_mode = "noshrink"
        native_dynamic, native_dynamic_details = model._inject(
            z, actions=actions, return_memory_details=True)
        memory.v7_mode = old_mode
        assert torch.equal(dynamic, native_dynamic)
        assert torch.equal(dynamic_details["states"], native_dynamic_details["states"])

        mixed, mixed_details = analysis.memory_with_rho(memory, z, actions, (0.0, 1.0))
        assert torch.equal(mixed_details["gates"][:, :, 0], static_details["gates"][:, :, 0])
        assert torch.equal(mixed_details["gates"][:, :, 1], dynamic_details["gates"][:, :, 1])
        assert not torch.equal(mixed, static)
        assert not torch.equal(mixed, dynamic)


def test_prediction_stage_has_no_target_dependency():
    model = make_model()
    observed, actions = synthetic_batch()
    predictions, diagnostics = analysis.predict_conditions(model, observed, actions, batch_size=2)
    assert tuple(predictions) == analysis.CONDITION_ORDER
    assert diagnostics["native_recurrence_max_abs"] == 0.0
    before = {name: value.clone() for name, value in predictions.items()}

    torch.manual_seed(31)
    targets_a = torch.randn(4, 32, 8, dtype=torch.float32)
    targets_b = targets_a + 2.0
    _, scores_a = analysis.score_predictions(predictions, targets_a, history_len=3)
    _, scores_b = analysis.score_predictions(predictions, targets_b, history_len=3)
    assert scores_a["learned"]["mse_first_post"] != scores_b["learned"]["mse_first_post"]
    for name in analysis.CONDITION_ORDER:
        assert torch.equal(predictions[name], before[name])


def test_observed_cache_loader_never_opens_clean_target():
    class GuardedCache:
        files = [
            "schema_version", "split", "clean_env", "occ_env", "features_input",
            "features_target", "actions", "target_valid_mask", "n_actions",
            "constant_target", "feature_dim", "manifest_sha256",
        ]

        def __init__(self, manifest_sha):
            self.values = {
                "schema_version": np.asarray(1), "split": np.asarray("val"),
                "manifest_sha256": np.asarray(manifest_sha), "n_actions": np.asarray(6),
                "feature_dim": np.asarray(128),
                "features_input": np.zeros((150, 32, 128), dtype=np.float32),
                "actions": np.zeros((150, 31), dtype=np.int64),
                "target_valid_mask": np.asarray(
                    [not 10 <= time < 16 for time in range(32)], dtype=np.bool_),
            }

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __getitem__(self, name):
            if name == "features_target":
                raise AssertionError("clean target was opened before prediction")
            return self.values[name]

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        cache_path = root / "val.npz"
        manifest_path = root / "manifest.json"
        cache_path.write_bytes(b"guarded-cache-placeholder")
        manifest_path.write_text("{}\n")
        guarded = GuardedCache(analysis.sha256_file(manifest_path))
        original_load = analysis.np.load
        analysis.np.load = lambda *_args, **_kwargs: guarded
        try:
            observed, actions, n_actions = analysis.load_validation_inputs(
                cache_path, manifest_path)
        finally:
            analysis.np.load = original_load
        assert observed.shape == (150, 32, 128)
        assert actions.shape == (150, 31)
        assert n_actions == 6


def test_phase_protocol_is_exact():
    phases = analysis.phase_indices(length=32, history_len=3)
    target_times = np.arange(3, 32)
    expected = {
        "pre": np.arange(3, 10),
        "blackout_transition": np.arange(10, 13),
        "deep_blackout": np.arange(13, 16),
        "first_post": np.asarray([16]),
        "recovery": np.arange(17, 19),
        "late_post": np.arange(19, 32),
        "all": np.arange(3, 32),
    }
    for phase, wanted_times in expected.items():
        assert np.array_equal(target_times[phases[phase]], wanted_times)


def test_manifest_records_fail_closed_and_official_inputs_validate():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "artifact.bin"
        path.write_bytes(b"endpoint-test")
        record = analysis.file_record(path)
        assert analysis.verify_file_record(path, record, "synthetic") == record
        path.write_bytes(b"tampered")
        try:
            analysis.verify_file_record(path, record, "synthetic")
        except analysis.EndpointReplayError as exc:
            assert "artifact mismatch" in str(exc)
        else:
            raise AssertionError("tampered artifact record was accepted")

    manifest, protocol, snapshot = analysis.validate_sealed_study(
        analysis.SEALED_ROOT, analysis.OFFICIAL_MANIFEST_SHA256)
    assert manifest["completed_runs"] == 325
    assert protocol["producer_git_commit"] == manifest["producer_git_commit"]
    assert analysis.input_path_key(
        analysis.SEALED_ROOT / "hacssm_v7_manifest.json") in snapshot
    assert analysis.input_path_key(
        analysis.SEALED_ROOT / "hacssm_v7_manifest.sha256") in snapshot
    assert set(manifest["source_artifacts"]).issubset(snapshot)
    assert set(manifest["feature_artifacts"]).issubset(snapshot)
    assert set(manifest["eval_rollout_artifacts"]).issubset(snapshot)
    validator_source = inspect.getsource(analysis.validate_sealed_study)
    assert "rev-parse" not in validator_source
    assert "producer_git_commit" in validator_source


def complete_run_row(condition: str = "learned") -> dict:
    rho = (0.4, 0.6) if condition == "learned" else analysis.FIXED_RHOS[condition]
    row = {
        "run_name": "synthetic", "env": "synthetic.occ", "seed": 0,
        "condition": condition, "rho_fast": rho[0], "rho_medium": rho[1],
        "episodes": 150, "length": 32, "history_len": 3,
        "checkpoint_sha256": "1" * 64, "val_feature_sha256": "2" * 64,
        "native_recurrence_max_abs": 0.0, "static_gate_fast": 0.5,
        "static_gate_medium": 0.5, "route_fast": 0.5, "route_medium": 0.5,
        "action_head_fast_norm": 1.0, "action_head_medium_norm": 1.0,
        "action_head_cosine": 0.25,
    }
    row.update({f"mse_{phase}": 1.0 for phase in analysis.PHASE_ORDER})
    return row


def complete_episode_row(condition: str = "learned") -> dict:
    row = {
        "run_name": "synthetic", "env": "synthetic.occ", "seed": 0,
        "episode": 0, "condition": condition, "rho_fast": 0.4, "rho_medium": 0.6,
    }
    row.update({f"mse_{phase}": 1.0 for phase in analysis.PHASE_ORDER})
    return row


def test_transactional_sibling_publication():
    with tempfile.TemporaryDirectory() as directory:
        output_parent = Path(directory)
        consumed = output_parent / "consumed.bin"
        consumed.write_bytes(b"immutable-input")
        input_snapshot = {analysis.input_path_key(consumed): analysis.file_record(consumed)}
        sealed_sha = "a" * 64
        manifest = {
            "producer_git_commit": "b" * 40, "completed_runs": 325,
            "protocol": {"protocol.json": {"bytes": 1, "sha256": "c" * 64}},
            "source_artifacts": {}, "feature_artifacts": {}, "eval_rollout_artifacts": {},
        }
        protocol = {"common_protocol": {"history_len": 3, "length": 32, "val_episodes": 150}}
        summary = {"native_recurrence_max_abs": 0.0}
        final = analysis.publish_results(
            output_parent, sealed_sha, analysis.SEALED_ROOT, manifest, protocol,
            [complete_run_row()], [complete_episode_row()], summary, [], input_snapshot,
            batch_size=64)
        assert final.name == "hacssm_v7_endpoints_aaaaaaaaaaaa"
        own_manifest = json.loads((final / "manifest.json").read_text())
        for name, record in own_manifest["outputs"].items():
            assert analysis.file_record(final / name) == record
        assert b"\r" not in (final / "endpoint_per_run.csv").read_bytes()
        assert b"\r" not in (final / "endpoint_per_episode.csv").read_bytes()
        sidecar = (final / "manifest.sha256").read_text()
        assert sidecar == f"{analysis.sha256_file(final / 'manifest.json')}  manifest.json\n"
        assert not list(output_parent.glob(".*.tmp"))
        try:
            analysis.publish_results(
                output_parent, sealed_sha, analysis.SEALED_ROOT, manifest, protocol,
                [complete_run_row()], [complete_episode_row()], summary, [], input_snapshot,
                batch_size=64)
        except analysis.EndpointReplayError as exc:
            assert "refusing to overwrite" in str(exc)
        else:
            raise AssertionError("transactional publisher overwrote an existing result")


def test_publication_rehash_rejects_consumed_input_mutation():
    with tempfile.TemporaryDirectory() as directory:
        output_parent = Path(directory)
        consumed = output_parent / "consumed.bin"
        consumed.write_bytes(b"before-replay")
        input_snapshot = {analysis.input_path_key(consumed): analysis.file_record(consumed)}
        manifest = {
            "producer_git_commit": "b" * 40, "completed_runs": 325,
            "protocol": {"protocol.json": {"bytes": 1, "sha256": "c" * 64}},
            "source_artifacts": {}, "feature_artifacts": {}, "eval_rollout_artifacts": {},
        }
        protocol = {
            "common_protocol": {"history_len": 3, "length": 32, "val_episodes": 150}}
        summary = {"native_recurrence_max_abs": 0.0}
        original_write_json = analysis.write_json_file

        def mutate_after_manifest_write(path, value):
            original_write_json(path, value)
            if path.name == "manifest.json":
                consumed.write_bytes(b"mutated-before-publication")

        analysis.write_json_file = mutate_after_manifest_write
        try:
            try:
                analysis.publish_results(
                    output_parent, "d" * 64, analysis.SEALED_ROOT, manifest, protocol,
                    [complete_run_row()], [complete_episode_row()], summary, [], input_snapshot,
                    batch_size=64)
            except analysis.EndpointReplayError as exc:
                assert "post-replay input" in str(exc) and "artifact mismatch" in str(exc)
            else:
                raise AssertionError("publication accepted a mutated consumed input")
        finally:
            analysis.write_json_file = original_write_json
        assert not (output_parent / "hacssm_v7_endpoints_dddddddddddd").exists()
        assert not list(output_parent.glob(".*.tmp"))


def test_summary_uses_25_paired_cells():
    rows = []
    for env_index in range(5):
        for seed in range(5):
            for condition in analysis.CONDITION_ORDER:
                row = complete_run_row(condition)
                row["run_name"] = f"env{env_index}-s{seed}"
                row["env"] = f"env{env_index}.occ"
                row["seed"] = seed
                base = 1.0 if condition == "learned" else 1.25
                row.update({f"mse_{phase}": base for phase in analysis.PHASE_ORDER})
                rows.append(row)
    summary = analysis.summarize(rows)
    assert summary["cells"] == 25
    assert summary["native_recurrence_max_abs"] == 0.0
    contrast = summary["paired_contrasts_vs_learned"]["rho00"]["first_post"]
    assert np.isclose(contrast["mean_paired_relative_learned_advantage"], 0.2)
    assert contrast["learned_wins"] == 25
    assert summary["learned_vs_joint_endpoint_envelope"]["first_post"]["learned_wins"] == 25
    encoded = json.dumps(summary, sort_keys=True)
    assert "mean_cell_mse" not in encoded
    assert "median_cell_mse" not in encoded
    for condition in analysis.CONDITION_ORDER:
        for phase in analysis.PHASE_ORDER:
            assert set(summary["conditions_summary"][condition][phase]) == {
                "environment_means"}


if __name__ == "__main__":
    tests = (
        test_replay_matches_native_and_exact_endpoints,
        test_prediction_stage_has_no_target_dependency,
        test_observed_cache_loader_never_opens_clean_target,
        test_phase_protocol_is_exact,
        test_manifest_records_fail_closed_and_official_inputs_validate,
        test_transactional_sibling_publication,
        test_publication_rehash_rejects_consumed_input_mutation,
        test_summary_uses_25_paired_cells,
    )
    for test in tests:
        test()
        print(f"{test.__name__}: OK")
    print(f"All {len(tests)} HACSSM-v7 endpoint analyzer tests passed.")
