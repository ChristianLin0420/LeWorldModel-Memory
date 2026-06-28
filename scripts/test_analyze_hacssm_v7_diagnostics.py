#!/usr/bin/env python3
"""Unit and frozen-input tests for the HACSSM-v7 post-hoc analyzer."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_hacssm_v7_diagnostics as diagnostics


class DiagnosticsTest(unittest.TestCase):
    def test_percentile_linear_interpolation(self) -> None:
        self.assertEqual(diagnostics.percentile([1.0, 2.0, 3.0], 0.5), 2.0)
        self.assertAlmostEqual(diagnostics.percentile([0.0, 1.0], 0.95), 0.95)

    def test_paired_summary_preserves_pairing(self) -> None:
        lookup = {}
        for env, candidate, reference in (
            ("a", (1.0, 6.0), (2.0, 4.0)),
            ("b", (3.0, 2.0), (3.0, 4.0)),
        ):
            for seed in (0, 1):
                lookup[(env, "candidate", seed)] = {"metric": str(candidate[seed])}
                lookup[(env, "reference", seed)] = {"metric": str(reference[seed])}
        observed = diagnostics.paired_summary(
            lookup, "candidate", "reference", metric="metric",
            envs=("a", "b"), seeds=(0, 1))
        wanted = ((2 - 1) / 2 + (4 - 6) / 4 + (3 - 3) / 3 + (4 - 2) / 4) / 4
        self.assertAlmostEqual(observed["mean_paired_relative_reduction"], wanted)
        self.assertEqual(observed["paired_wins"], 2)
        self.assertEqual(observed["paired_ties"], 1)

    def test_rank_rows_are_tie_safe(self) -> None:
        old_envs, old_designs, old_seeds = (
            diagnostics.ENVIRONMENTS, diagnostics.DESIGNS, diagnostics.SEEDS)
        try:
            diagnostics.ENVIRONMENTS = ("env",)
            diagnostics.DESIGNS = ("a", "b", "c")
            diagnostics.SEEDS = (0, 1)
            lookup = {
                ("env", "a", 0): {diagnostics.PRIMARY: "1"},
                ("env", "a", 1): {diagnostics.PRIMARY: "1"},
                ("env", "b", 0): {diagnostics.PRIMARY: "1"},
                ("env", "b", 1): {diagnostics.PRIMARY: "1"},
                ("env", "c", 0): {diagnostics.PRIMARY: "2"},
                ("env", "c", 1): {diagnostics.PRIMARY: "2"},
            }
            env_rows, cell_rows = diagnostics.rank_rows(lookup)
            env_rank = {row["design"]: row["rank"] for row in env_rows}
            self.assertEqual(env_rank, {"a": 1, "b": 1, "c": 3})
            seed_zero = {row["design"]: row["rank"] for row in cell_rows if row["seed"] == 0}
            self.assertEqual(seed_zero, {"a": 1, "b": 1, "c": 3})
        finally:
            diagnostics.ENVIRONMENTS, diagnostics.DESIGNS, diagnostics.SEEDS = (
                old_envs, old_designs, old_seeds)

    def test_overall_contrasts_do_not_pool_raw_mse(self) -> None:
        old_envs, old_designs, old_seeds = (
            diagnostics.ENVIRONMENTS, diagnostics.DESIGNS, diagnostics.SEEDS)
        try:
            diagnostics.ENVIRONMENTS = ("a", "b")
            diagnostics.DESIGNS = ("candidate", "reference")
            diagnostics.SEEDS = old_seeds
            lookup = {}
            for seed in old_seeds:
                lookup[("a", "candidate", seed)] = {diagnostics.PRIMARY: "1"}
                lookup[("a", "reference", seed)] = {diagnostics.PRIMARY: "2"}
                lookup[("b", "candidate", seed)] = {diagnostics.PRIMARY: "100"}
                lookup[("b", "reference", seed)] = {diagnostics.PRIMARY: "200"}
            rows = diagnostics.contrast_rows(lookup)
            overall = diagnostics.find_overall(rows, "candidate", "reference")
            self.assertEqual(overall["candidate_mean_mse"], "")
            self.assertEqual(overall["reference_mean_mse"], "")
            self.assertEqual(overall["mean_paired_relative_reduction"], 0.5)
        finally:
            diagnostics.ENVIRONMENTS, diagnostics.DESIGNS, diagnostics.SEEDS = (
                old_envs, old_designs, old_seeds)

    def test_state_and_rollout_difference(self) -> None:
        state = {"x": torch.tensor([1.0, 2.0]), "n": torch.tensor([2])}
        exact, maximum = diagnostics.tensor_state_difference(state, state)
        self.assertTrue(exact)
        self.assertEqual(maximum, 0.0)
        changed = {"x": torch.tensor([1.0, 5.0]), "n": torch.tensor([2])}
        exact, maximum = diagnostics.tensor_state_difference(state, changed)
        self.assertFalse(exact)
        self.assertEqual(maximum, 3.0)
        arrays = {"x": np.asarray([1.0, 2.0], dtype=np.float32)}
        changed_arrays = {"x": np.asarray([1.0, 3.0], dtype=np.float32)}
        exact, maximum = diagnostics.rollout_difference(arrays, changed_arrays)
        self.assertFalse(exact)
        self.assertEqual(maximum, 1.0)

    def test_history_summary_recognizes_99_active_epochs(self) -> None:
        history = []
        for epoch in range(1, 201):
            weight = 0.02 if epoch <= 40 else (0.01 if epoch < 100 else 0.0)
            values = {
                "loss": 1.0, "pred_loss": 0.8, "sigreg_loss": 0.2,
                "hier_loss": 0.5, "hier_loss_weight": weight,
                "hier_loss_fast": 0.4, "hier_loss_medium": 0.6,
                "hier_loss_bridge": 0.45, "hier_loss_recovery": 0.55,
                "hier_overlap": 0.0,
            }
            history.append({"epoch": epoch, "train": dict(values), "val": dict(values)})
        observed = diagnostics.history_summary(history, "fixture")
        self.assertEqual(observed["active_aux_epochs"], 99)
        self.assertEqual(observed["active_train_hier_loss_mean"], 0.5)

    def test_artifact_reader_rejects_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(dir=diagnostics.REPO_ROOT) as directory:
            root = Path(directory)
            path = root / "artifact.json"
            path.write_text('{"a": 1}\n')
            relative = path.absolute().relative_to(diagnostics.REPO_ROOT).as_posix()
            manifest = {"output_artifacts": {
                relative: {"bytes": path.stat().st_size, "sha256": "0" * 64}}}
            reader = diagnostics.ArtifactReader(root, manifest)
            with self.assertRaisesRegex(ValueError, "hash differs"):
                reader.verify(path)

    def test_json_writer_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            one, two = root / "one.json", root / "two.json"
            value = {"z": [2, 1], "a": {"b": True}}
            diagnostics.write_json(one, value)
            diagnostics.write_json(two, value)
            self.assertEqual(one.read_bytes(), two.read_bytes())

    def test_frozen_manifest_and_decisions_recompute(self) -> None:
        manifest = diagnostics.verify_manifest(
            diagnostics.V7_ROOT, "hacssm_v7_manifest.json",
            diagnostics.V7_MANIFEST_SHA256, diagnostics.V7_PRODUCER_COMMIT, "v7")
        self.assertEqual(manifest["completed_runs"], 325)
        rows = diagnostics.load_csv(diagnostics.V7_ROOT / "per_run.csv")
        diagnostics.validate_rows(rows)
        convergence = diagnostics.load_csv(diagnostics.V7_ROOT / "convergence.csv")
        pilot, final = diagnostics.recompute_decisions(rows, convergence, diagnostics.V7_ROOT)
        self.assertEqual(pilot["decision"], "NO_GO")
        self.assertEqual(final["decision"], "PILOT_NO_GO_FINAL_DESCRIPTIVE")


if __name__ == "__main__":
    unittest.main()
