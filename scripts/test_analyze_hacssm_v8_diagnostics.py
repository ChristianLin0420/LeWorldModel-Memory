#!/usr/bin/env python3
"""Synthetic and sealed-input tests for the HACSSM-v8 post-hoc analyzer."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_hacssm_v8_diagnostics as diagnostics


class DiagnosticsTest(unittest.TestCase):
    def test_paired_summary_preserves_pairs_without_raw_pooling(self) -> None:
        lookup = {}
        fixtures = {
            "a": ((1.0, 6.0), (2.0, 4.0)),
            "b": ((3.0, 2.0), (3.0, 4.0)),
        }
        for env, (candidate, reference) in fixtures.items():
            for seed in (0, 1):
                lookup[(env, "candidate", seed)] = {"metric": str(candidate[seed])}
                lookup[(env, "reference", seed)] = {"metric": str(reference[seed])}
        observed = diagnostics.paired_summary(
            lookup, "candidate", "reference", metric="metric",
            environments=("a", "b"), seeds=(0, 1),
        )
        expected = ((2 - 1) / 2 + (4 - 6) / 4 + (3 - 3) / 3 + (4 - 2) / 4) / 4
        self.assertAlmostEqual(observed["mean_paired_relative_reduction"], expected)
        self.assertEqual(observed["paired_wins"], 2)
        self.assertEqual(observed["paired_ties"], 1)
        self.assertNotIn("candidate_mean_mse", observed)
        self.assertNotIn("reference_mean_mse", observed)

    def test_environment_rank_is_tie_safe(self) -> None:
        lookup = {}
        for design, value in (("a", 1.0), ("b", 1.0), ("c", 2.0)):
            for seed in (0, 1):
                lookup[("env", design, seed)] = {diagnostics.PRIMARY: str(value)}
        rows = diagnostics.environment_rank_rows(
            lookup, environments=("env",), designs=("a", "b", "c"), seeds=(0, 1)
        )
        self.assertEqual({row["design"]: row["rank"] for row in rows}, {"a": 1, "b": 1, "c": 3})

    def test_crossed_bootstrap_is_deterministic(self) -> None:
        matrix = np.asarray([[0.1, 0.2], [-0.1, 0.0], [0.3, 0.4]], dtype=np.float64)
        one = diagnostics.crossed_bootstrap(matrix, draws=2_000, seed=77)
        two = diagnostics.crossed_bootstrap(matrix, draws=2_000, seed=77)
        self.assertEqual(one, two)
        self.assertAlmostEqual(one["point_mean_paired_relative_reduction"], 0.15)
        self.assertLessEqual(one["ci95_low"], one["ci90_low"])
        self.assertGreaterEqual(one["ci95_high"], one["ci90_high"])
        self.assertEqual(
            diagnostics.canonical_json_sha256(diagnostics.BOOTSTRAP_CONTRACT),
            diagnostics.BOOTSTRAP_CONTRACT_SHA256,
        )

    def test_manifest_pair_fails_closed_on_sidecar_change(self) -> None:
        with tempfile.TemporaryDirectory(dir=diagnostics.REPO_ROOT) as directory:
            root = Path(directory)
            manifest = root / "hacssm_v8_manifest.json"
            diagnostics.write_json_new(manifest, {"schema_version": 1})
            digest = diagnostics.sha256_file(manifest)
            sidecar = root / "hacssm_v8_manifest.sha256"
            sidecar.write_text(f"{digest}  hacssm_v8_manifest.json\n")
            self.assertEqual(diagnostics.verify_manifest_pair(root, digest)["schema_version"], 1)
            sidecar.write_text(f"{'0' * 64}  hacssm_v8_manifest.json\n")
            with self.assertRaisesRegex(ValueError, "sidecar mismatch"):
                diagnostics.verify_manifest_pair(root, digest)

    def test_artifact_verifier_fails_closed_on_hash_and_symlink_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.txt"
            artifact.write_text("sealed\n")
            record = diagnostics.file_record(artifact)
            diagnostics.verify_artifact(artifact, record, "fixture")
            artifact.write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "differs from manifest"):
                diagnostics.verify_artifact(artifact, record, "fixture")

            target = root / "target"
            target.write_text("x")
            link = root / "link"
            link.symlink_to("target")
            diagnostics.verify_artifact(
                link, {"kind": "symlink", "target": "target"}, "link"
            )
            link.unlink()
            link.symlink_to("artifact.txt")
            with self.assertRaisesRegex(ValueError, "symlink differs"):
                diagnostics.verify_artifact(
                    link, {"kind": "symlink", "target": "target"}, "link"
                )

    def test_atomic_package_refuses_overwrite_and_has_valid_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            output = parent / "package"
            values = {
                name: ([{"value": 1}] if name.endswith(".csv") else {"value": 1})
                for name in diagnostics.OUTPUT_FILES
            }
            diagnostics.publish_package(
                output, values, primary_input_hashes={"fixture": "abc"},
                reverify=lambda: None,
            )
            digest = diagnostics.sha256_file(output / "posthoc_manifest.json")
            self.assertEqual(
                (output / "posthoc_manifest.sha256").read_text(),
                f"{digest}  posthoc_manifest.json\n",
            )
            manifest = json.loads((output / "posthoc_manifest.json").read_text())
            self.assertFalse(manifest["raw_pca_mse_pooled_across_environments"])
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                diagnostics.publish_package(
                    output, values, primary_input_hashes={"fixture": "abc"},
                    reverify=lambda: None,
                )

    def test_frozen_hashes_grid_arithmetic_and_parameters(self) -> None:
        required = [
            diagnostics.PRIMARY_ROOT / "hacssm_v8_manifest.json",
            diagnostics.PRIMARY_ROOT / "hacssm_v8_manifest.sha256",
            *(
                diagnostics.PRIMARY_ROOT
                / diagnostics.run_name(env, diagnostics.CANDIDATE, seed)
                / "model.pt"
                for env in diagnostics.ENVIRONMENTS
                for seed in diagnostics.SEEDS
            ),
        ]
        if not all(path.is_file() for path in required):
            self.skipTest("requires the sealed V8 primary artifact and checkpoint tree")
        verified = diagnostics.verify_primary(full_artifact_audit=False)
        self.assertEqual(
            diagnostics.sha256_file(diagnostics.PRIMARY_ROOT / "decision.json"),
            diagnostics.FINAL_DECISION_SHA256,
        )
        self.assertEqual(
            diagnostics.sha256_file(diagnostics.PRIMARY_ROOT / "equivalence_receipts.json"),
            diagnostics.EQUIVALENCE_RECEIPTS_SHA256,
        )
        rows = diagnostics.load_csv(diagnostics.PRIMARY_ROOT / "per_run.csv")
        lookup = diagnostics.validate_rows(rows)
        ranking = diagnostics.ssm_ranking_rows(lookup)
        candidate = next(row for row in ranking if row["design"] == diagnostics.CANDIDATE)
        leader = next(row for row in ranking if row["design"] == diagnostics.V7_LEADER)
        self.assertEqual(candidate["rank"], 2)
        self.assertEqual(leader["rank"], 1)
        self.assertAlmostEqual(candidate["mean_paired_relative_reduction"], 0.0620356407397219)

        convergence = diagnostics.convergence_rows(
            diagnostics.load_csv(diagnostics.PRIMARY_ROOT / "convergence.csv")
        )
        bootstraps = diagnostics.bootstrap_rows(lookup)
        diagnostics.validate_locked_arithmetic(
            lookup, convergence, bootstraps, verified["pilot"], verified["decision"]
        )
        final_ni = next(
            row for row in bootstraps
            if row["stage"] == "final"
            and row["comparison_kind"] == "v7_leader_noninferiority"
        )
        self.assertAlmostEqual(final_ni["ci95_low"], -0.01260065880004873)

        parameters = diagnostics.learned_parameter_rows(
            diagnostics.PRIMARY_ROOT, lookup, verified["manifest"]
        )
        summary = diagnostics.parameter_summary(parameters)
        self.assertEqual(summary["n_runs"], 25)
        self.assertAlmostEqual(summary["rho_fast"]["mean"], 0.6198841735358191)
        self.assertAlmostEqual(summary["route_fast"]["mean"], 0.5563943765318804)


if __name__ == "__main__":
    unittest.main()
