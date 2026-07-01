#!/usr/bin/env python3
"""Focused deterministic tests for the independent Sub-JEPA-v16 closeout."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.closeout_subjepa_v16 import (
    CLEAN_METRIC,
    CONTRAST_METRICS,
    DESIGNS,
    FAMILIES,
    MEMORIES,
    PRIMARY_METRIC,
    SEEDS,
    TASKS,
    VAL_PREDICTIVE_METRIC,
    _history_remote_rows,
    _validate_projection,
    _validate_history,
    audit_attempt_ledgers,
    build_interactions,
    close_enough,
    expected_metadata,
    json_sha256,
    local_error_events,
    output_path_for,
    paired_contrast,
    remote_config_matches,
    remote_media_matches,
    seed_block_summary,
    write_exclusive,
)


class CloseoutMathTests(unittest.TestCase):
    def test_seed_blocked_interval_uses_three_seed_blocks(self) -> None:
        tasks = ("task-a", "task-b")
        seed_effect = {SEEDS[0]: 0.1, SEEDS[1]: 0.2, SEEDS[2]: 0.3}
        effects = {
            (task, seed): seed_effect[seed]
            for task in tasks
            for seed in SEEDS
        }
        result = seed_block_summary(effects, tasks=tasks, seeds=SEEDS)
        self.assertEqual(result["n_task_seed_pairs"], 6)
        self.assertEqual(result["n_seed_blocks"], 3)
        self.assertAlmostEqual(result["point_estimate"], 0.2)
        self.assertAlmostEqual(result["seed_block_std"], 0.1)
        expected_half_width = result["t_critical"] * 0.1 / math.sqrt(3)
        self.assertAlmostEqual(result["ci95_low"], 0.2 - expected_half_width)
        self.assertAlmostEqual(result["ci95_high"], 0.2 + expected_half_width)
        self.assertFalse(result["raw_cross_task_ratio_of_means_used"])

    def test_paired_effect_is_mean_of_within_cell_relatives_not_pooled_ratio(self) -> None:
        metrics = {}
        expected_effects = []
        for task_index, task in enumerate(TASKS):
            for seed_index, seed in enumerate(SEEDS):
                reference = 1000.0 if task_index == 0 else 1.0
                effect = 0.01 * (task_index + 1) + 0.001 * seed_index
                candidate = reference * (1.0 - effect)
                metrics[(task, "subjepa16_none", seed)] = {PRIMARY_METRIC: candidate}
                metrics[(task, "fullsig_none", seed)] = {PRIMARY_METRIC: reference}
                expected_effects.append(effect)
        result = paired_contrast(
            metrics,
            comparison="synthetic",
            candidate_design="subjepa16_none",
            reference_design="fullsig_none",
            metric=PRIMARY_METRIC,
        )
        expected = sum(expected_effects) / len(expected_effects)
        pooled_candidate = sum(
            row[PRIMARY_METRIC]
            for key, row in metrics.items()
            if key[1] == "subjepa16_none"
        )
        pooled_reference = sum(
            row[PRIMARY_METRIC]
            for key, row in metrics.items()
            if key[1] == "fullsig_none"
        )
        pooled_ratio = (pooled_reference - pooled_candidate) / pooled_reference
        self.assertAlmostEqual(result["point_estimate"], expected)
        self.assertNotAlmostEqual(result["point_estimate"], pooled_ratio, places=4)

    def test_host_memory_interaction_uses_difference_of_log_benefits(self) -> None:
        metrics = {}
        for task in TASKS:
            for seed in SEEDS:
                for design in DESIGNS:
                    metrics[(task, design, seed)] = {
                        metric: 1.0 for metric in CONTRAST_METRICS
                    }
                for metric in CONTRAST_METRICS:
                    metrics[(task, "fullsig_none", seed)][metric] = 1.0
                    metrics[(task, "subjepa16_none", seed)][metric] = 0.8
                    metrics[(task, "fullsig_ssm", seed)][metric] = 1.0
                    metrics[(task, "subjepa16_ssm", seed)][metric] = 0.4
        interactions = build_interactions(metrics)
        self.assertEqual(len(interactions), 18)
        target = next(
            row for row in interactions
            if row["interaction"]
            == f"subjepa16_vs_fullsig:ssm_minus_none:{PRIMARY_METRIC}"
        )
        self.assertAlmostEqual(target["point_estimate"], math.log(2.5) - math.log(1.25))
        self.assertGreater(target["point_estimate"], 0.0)


class LedgerAndArtifactTests(unittest.TestCase):
    @staticmethod
    def _ledger_row(key: tuple[str, str, int], argv: list[str]) -> dict[str, object]:
        task, design, seed = key
        return {
            "task": task,
            "design": design,
            "seed": seed,
            "status": "complete",
            "resumed_existing": False,
            "command_sha256": json_sha256(argv),
            "wandb_receipt_present": True,
            "artifact_sha256": {"model.pt": "abc"},
            "seconds": 1.0,
            "gpu": "0",
            "completed_at": "2026-01-01T00:00:00+00:00",
            "directory": "/tmp/cell",
            "log": "/tmp/cell.log",
        }

    def test_attempt_ledger_requires_one_clean_attempt_per_cell(self) -> None:
        keys = {
            ("task-a", "subjepa16_none", 1),
            ("task-b", "subjepa16_none", 1),
        }
        command_map = {}
        rows = []
        for key in sorted(keys):
            argv = ["python", "train.py", key[0]]
            command_map[key] = {"argv": argv}
            rows.append(self._ledger_row(key, argv))
        report, index, audit_rows = audit_attempt_ledgers(
            keys, command_map, list(rows), list(rows))
        self.assertTrue(report["passed"])
        self.assertTrue(report["exactly_one_attempt_per_cell"])
        self.assertFalse(report["result_dependent_relaunch_detected"])
        self.assertEqual(len(index), 2)
        self.assertEqual(len(audit_rows), 2)

        failed_prior = dict(rows[0])
        failed_prior["status"] = "failed"
        report, _, _ = audit_attempt_ledgers(
            keys, command_map, list(rows), [failed_prior, *rows])
        self.assertFalse(report["passed"])
        self.assertFalse(report["exactly_one_attempt_per_cell"])
        self.assertTrue(report["result_dependent_relaunch_detected"])

    def test_projection_tensor_shape_gram_quadrature_and_sha(self) -> None:
        subspaces = 16
        width = 8
        base = torch.eye(128, dtype=torch.float32)[:width]
        matrices = base.unsqueeze(0).repeat(subspaces, 1, 1).contiguous()
        knots = torch.linspace(0.0, 3.0, 17, dtype=torch.float32)
        phi = torch.exp(-knots.square() / 2.0)
        quadrature = torch.full((17,), 2.0 * (3.0 / 16.0), dtype=torch.float32)
        quadrature[[0, -1]] = 3.0 / 16.0
        weights = quadrature * phi
        digest = hashlib.sha256(
            matrices.numpy().astype("<f4", copy=False).tobytes(order="C")
        ).hexdigest()
        state = {
            "world.sigreg.projection_matrices": matrices,
            "world.sigreg.t": knots,
            "world.sigreg.phi": phi,
            "world.sigreg.weights": weights,
        }
        metrics = {
            "subspace_projection_frozen": True,
            "subspace_projection_count": subspaces,
            "subspace_projection_dimension": width,
            "subspace_projection_sha256": digest,
            # A CUDA-vs-CPU GEMM diagnostic may differ by a few FP32 ulps.  The
            # projection tensor itself is still checked by exact byte hash.
            "subspace_projection_orthogonality_max_abs": 3e-7,
        }
        result = _validate_projection(state, metrics, "subjepa16", "synthetic")
        self.assertEqual(result["projection_sha256"], digest)
        self.assertEqual(result["orthogonality_max_abs"], 0.0)

    def test_history_csv_has_one_paired_train_val_row_per_epoch(self) -> None:
        history = []
        for epoch in range(1, 31):
            split = {
                "loss": 0.3,
                "predictive_loss": 0.2,
                "regularizer_loss": 0.1,
                "sigreg_loss": 1.0,
                "variance_loss": 0.0,
                "covariance_loss": 0.0,
            }
            history.append({
                "epoch": epoch,
                "epoch_seconds": 1.0,
                "train": dict(split),
                "val": dict(split),
            })
        rows = _validate_history(history, "task-a", "subjepa16_none", 1)
        self.assertEqual(len(rows), 30)
        self.assertIn("train_predictive_loss", rows[0])
        self.assertIn("val_predictive_loss", rows[0])
        self.assertNotIn("split", rows[0])


class CreateOnlyTests(unittest.TestCase):
    def test_output_is_protocol_hash_keyed_and_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = root / "development_protocol.json"
            protocol.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
            output = output_path_for(protocol, root / "out")
            expected_hash = hashlib.sha256(protocol.read_bytes()).hexdigest()[:12]
            self.assertEqual(output.name, f"subjepa_v16_closeout_{expected_hash}")
            revised = output_path_for(protocol, root / "out", "r2")
            self.assertEqual(revised.name, f"subjepa_v16_closeout_{expected_hash}_r2")
            output.mkdir(parents=True)
            target = output / "receipt.txt"
            write_exclusive(target, "first\n")
            with self.assertRaises(FileExistsError):
                write_exclusive(target, "second\n")
            self.assertEqual(target.read_text(encoding="utf-8"), "first\n")

    def test_numeric_comparison_helper(self) -> None:
        self.assertTrue(close_enough(0.1 + 0.2, 0.3))
        self.assertFalse(close_enough("not-a-number", 0.0))

    def test_remote_config_maps_argparse_projection_name(self) -> None:
        expected = expected_metadata(
            "cartpole.swingup", "subjepa16_none", SEEDS[0])
        config = {
            remote_name: expected[receipt_name]
            for receipt_name, remote_name in {
                "design": "design",
                "seed": "seed",
                "env": "env",
                "regularizer": "regularizer",
                "regularizer_family": "regularizer_family",
                "num_subspaces": "num_subspaces",
                "subspace_dim": "subspace_dim",
                "memory_architecture": "memory_architecture",
                "regularizer_source": "regularizer_source",
                "clean_target_gradient_active": "clean_target_gradient_active",
                "target_stop_gradient": "target_stop_gradient",
                "sigreg_projections_per_subspace": "sigreg_projections",
                "sigreg_quad_nodes": "sigreg_quad_nodes",
            }.items()
        }
        config.update({
            "wandb_entity": "crlc112358",
            "wandb_project": "lewm-memory-popgym",
            "wandb_study": "subjepa-v16-development",
        })
        self.assertNotIn("sigreg_projections_per_subspace", config)
        self.assertTrue(remote_config_matches(config, expected))
        config["sigreg_projections"] = 511
        self.assertFalse(remote_config_matches(config, expected))

    def test_remote_errors_do_not_erase_local_integrity(self) -> None:
        events = [
            {"level": "error", "code": "remote_wandb"},
            {"level": "scientific_diagnostic", "code": "rank_gate_failed"},
        ]
        self.assertEqual(local_error_events(events), [])
        events.append({"level": "error", "code": "artifact_sha256"})
        self.assertEqual(len(local_error_events(events)), 1)

    def test_remote_history_uses_complete_history_api(self) -> None:
        class Run:
            def history(self, **kwargs: object) -> list[dict[str, int]]:
                self.kwargs = kwargs
                return [{"epoch": epoch} for epoch in range(1, 31)]

        run = Run()
        rows = _history_remote_rows(run)
        self.assertEqual([row["epoch"] for row in rows], list(range(1, 31)))
        self.assertEqual(run.kwargs["samples"], 10_000)
        self.assertFalse(run.kwargs["pandas"])

    def test_remote_media_uses_summary_receipts_and_table_artifact(self) -> None:
        class Artifact:
            type = "run_table"

        class SummaryChild:
            def __init__(self, value: dict[str, object]) -> None:
                self.value = value

            def items(self):
                return self.value.items()

        summary = {
            "eval/rollout_trace": {
                "_type": "table-file", "nrows": 180, "sha256": "table-sha",
            },
            "eval/paired_rollout": {
                "_type": "video-file", "size": 1024, "sha256": "video-sha",
            },
        }
        self.assertEqual(remote_media_matches(summary, [Artifact()]), (True, True))
        wrapped = {key: SummaryChild(value) for key, value in summary.items()}
        self.assertEqual(remote_media_matches(wrapped, [Artifact()]), (True, True))
        self.assertEqual(remote_media_matches(summary, []), (False, True))


if __name__ == "__main__":
    unittest.main()
