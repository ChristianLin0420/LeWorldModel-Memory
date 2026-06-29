#!/usr/bin/env python3
"""Focused unit tests for the independent SIRO-v12 screen auditor."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import audit_siro_v12_screen as audit


def _rollout_arrays(action_dim: int = 2, target_dim: int = 3) -> dict[str, np.ndarray]:
    length = 48
    target_times = np.arange(3, length, dtype=np.int64)
    count = len(target_times)
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(2, dtype=np.int64),
        "episode_index": np.asarray(0, dtype=np.int64),
        "conditions": np.asarray(audit.CONDITIONS),
    }
    flat: dict[str, list[np.ndarray]] = {
        "condition": [], "target_times": [], "phase": [], "state_target": []}
    for coordinate in audit.COORDINATES:
        flat[f"{coordinate}_state_prediction"] = []
        flat[f"{coordinate}_state_nmse"] = []
    for condition_index, condition in enumerate(audit.CONDITIONS):
        prefix = f"{condition}_"
        phase = np.asarray(
            ["context"] * 8 + ["gap"] * 8 + ["deep"] * 8
            + ["first_post"] + ["post"] * (count - 25))
        target = np.full((count, target_dim), condition_index + 0.5, dtype=np.float32)
        arrays.update({
            prefix + "target_times": target_times.copy(),
            prefix + "phase": phase,
            prefix + "gap_start": np.asarray(11, dtype=np.int64),
            prefix + "gap_end": np.asarray(27, dtype=np.int64),
            prefix + "observed_rgb": np.full(
                (length, 64, 64, 3), condition_index, dtype=np.uint8),
            prefix + "clean_rgb": np.full(
                (length, 64, 64, 3), condition_index + 1, dtype=np.uint8),
            prefix + "actions": np.full(
                (length - 1, action_dim), condition_index + 0.25, dtype=np.float32),
            prefix + "evaluation_target": target,
        })
        flat["condition"].append(np.full(count, condition))
        flat["target_times"].append(target_times.copy())
        flat["phase"].append(phase)
        flat["state_target"].append(target)
        for coordinate_index, coordinate in enumerate(audit.COORDINATES):
            prediction = target + np.float32(coordinate_index / 10)
            nmse = np.full(count, coordinate_index / 100, dtype=np.float32)
            arrays[prefix + f"{coordinate}_state_prediction"] = prediction
            arrays[prefix + f"{coordinate}_state_nmse_by_target_t"] = nmse
            flat[f"{coordinate}_state_prediction"].append(prediction)
            flat[f"{coordinate}_state_nmse"].append(nmse)
    arrays.update({key: np.concatenate(values) for key, values in flat.items()})
    return arrays


class GridTests(unittest.TestCase):
    def test_expected_grid_is_exactly_twenty_eight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = audit.expected_run_directories(root)
            self.assertEqual(len(expected), 28)
            for path in expected.values():
                path.mkdir()
            audit.validate_root_directory_set(root)
            (root / "unexpected").mkdir()
            with self.assertRaises(audit.AuditFailure):
                audit.validate_root_directory_set(root)

    def test_comparator_directory_encodes_ranking(self) -> None:
        name = audit.run_name("fish.swim", "kdiov11")
        self.assertIn("rank-rawdiff_displacement_detached", name)

    def test_metrics_schema_is_design_specific(self) -> None:
        self.assertEqual(audit.expected_metrics_schema_version("sirov12"), 1)
        self.assertEqual(audit.expected_metrics_schema_version("kdiov11"), 2)


class RepresentationReceiptTests(unittest.TestCase):
    def test_exact_seven_anchor_rank_failures_are_recomputed(self) -> None:
        rows = []
        for task in audit.TASKS:
            for design in audit.DESIGNS:
                rank = None
                if design in audit.SIRO_DESIGNS:
                    rank = 20.0
                    if task == "cartpole.swingup":
                        rank = 1.0
                    if task == "pendulum.swingup" and design == "sirov12_noanchor":
                        rank = 13.5
                rows.append({
                    "task": task,
                    "design": design,
                    "anchor_covariance_effective_rank": rank,
                })
        failures = audit.representation_rank_failures(rows)
        self.assertEqual(len(failures), 7)
        self.assertEqual(
            failures[-1],
            "pendulum.swingup/sirov12_noanchor: "
            "fit anchor effective rank below 16")

    def test_frozen_negative_analyzer_receipt_is_consistent(self) -> None:
        failures = [f"failure-{index}" for index in range(7)]
        protocol = {
            "scope": "excluded_adaptive_v12_screen_after_failed_v11",
            "study": "test-study",
            "epochs": 30,
        }
        analysis = {
            "schema_version": 1,
            "scope": protocol["scope"],
            "study": protocol["study"],
            "seed": audit.SEED,
            "epochs": 30,
            "expected_cells": 28,
            "completed_cells": 21,
            "integrity_passed": False,
            "integrity_errors": failures,
            "official_result": False,
            "iclr_confirmation": False,
            "status": "INCOMPLETE_OR_INVALID",
            "continue_to_100_epochs": False,
            "scientific_gate_passed": False,
        }
        decision = {
            "status": analysis["status"],
            "integrity_passed": False,
            "continue_to_100_epochs": False,
            "scientific_gate_passed": False,
            "automatic_launch_performed": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "screen_analysis.json").write_text(json.dumps(analysis))
            (root / "screen_decision.json").write_text(json.dumps(decision))
            self.assertEqual(
                audit.validate_analyzer_output(root, [], protocol, failures),
                "INCOMPLETE_OR_INVALID")


class RolloutTests(unittest.TestCase):
    def _write(self, root: Path, arrays: dict[str, np.ndarray]) -> tuple[Path, dict]:
        path = root / "eval_rollout.npz"
        np.savez_compressed(path, **arrays)
        metrics = {
            "eval_rollout_sha256": audit.sha256_file(path),
            "eval_rollout_episode": 0,
            "length": 48,
            "action_dim": 2,
            "eval_target_dim": 3,
        }
        return path, metrics

    def test_complete_rollout_schema_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, metrics = self._write(Path(temporary), _rollout_arrays())
            self.assertEqual(
                audit.validate_rollout(path, metrics, "synthetic"),
                metrics["eval_rollout_sha256"])

    def test_flat_trace_tampering_fails_even_with_matching_file_hash(self) -> None:
        arrays = _rollout_arrays()
        arrays["prior_state_prediction"] = arrays["prior_state_prediction"].copy()
        arrays["prior_state_prediction"][0, 0] += 1
        with tempfile.TemporaryDirectory() as temporary:
            path, metrics = self._write(Path(temporary), arrays)
            with self.assertRaisesRegex(audit.AuditFailure, "inconsistent"):
                audit.validate_rollout(path, metrics, "synthetic")

    def test_nonfinite_condition_array_fails(self) -> None:
        arrays = _rollout_arrays()
        arrays["freeze_actions"] = arrays["freeze_actions"].copy()
        arrays["freeze_actions"][0, 0] = np.nan
        with tempfile.TemporaryDirectory() as temporary:
            path, metrics = self._write(Path(temporary), arrays)
            with self.assertRaisesRegex(audit.AuditFailure, "non-finite"):
                audit.validate_rollout(path, metrics, "synthetic")


if __name__ == "__main__":
    unittest.main()
