#!/usr/bin/env python3
"""Protocol and orchestration tests for the sealed ORBIT-v10 study."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.run_hacssm_v10 as runner


class V10ProtocolTests(unittest.TestCase):
    def test_grid_is_exact_and_staged(self):
        self.assertEqual(len(runner.DESIGNS), 9)
        self.assertEqual(len(runner.ENVIRONMENTS), 5)
        self.assertEqual(len(runner.PILOT_JOBS), 135)
        self.assertEqual(len(runner.COMPLETION_JOBS), 90)
        self.assertEqual(len(runner.ALL_JOBS), 225)
        self.assertEqual(len({job.run_name for job in runner.ALL_JOBS}), 225)

    def test_native_end_to_end_common_contract(self):
        common = runner.COMMON
        self.assertEqual(common["train_episodes"], 1200)
        self.assertEqual(common["val_episodes"], 240)
        self.assertEqual(common["length"], 48)
        self.assertEqual(common["img_size"], 64)
        self.assertEqual(common["embed_dim"], 128)
        self.assertEqual(common["encoder_layers"], 6)
        self.assertEqual(common["predictor_layers"], 4)
        self.assertEqual(common["encoder_norm"], "none")
        self.assertEqual(common["predictor_norm"], "none")
        self.assertEqual(common["sigreg_lambda"], 0.1)

    def test_tasks_and_corruptions_are_prospectively_disjoint(self):
        self.assertEqual(
            tuple(environment for environment, _ in runner.ENVIRONMENTS),
            (
                "dmc:walker.walk", "dmc:hopper.hop", "dmc:cartpole.swingup",
                "dmc:pendulum.swingup", "dmc:fish.swim",
            ),
        )
        self.assertTrue(set(runner.TRAIN_CORRUPTIONS).isdisjoint(runner.HELDOUT_CORRUPTIONS))

    def test_data_paths_bind_counts_length_and_split_seed(self):
        train, val, manifest = runner.data_paths("dmc:walker.walk")
        self.assertEqual(
            train.name, "dmc_walker_walk_train_n1200_L48_s64_seed27100.npz"
        )
        self.assertEqual(
            val.name, "dmc_walker_walk_val_n240_L48_s64_seed92710.npz"
        )
        self.assertEqual(manifest.name, "manifest.json")

    def test_expected_args_have_no_checkpoint_or_feature_escape_hatch(self):
        job = runner.PILOT_JOBS[0]
        args = runner.expected_args(job)
        self.assertEqual(set(args), {
            "train_data", "val_data", "memory_mode", "seed", "output_dir", "epochs",
            "batch_size", "lr", "weight_decay", "num_workers", "wandb",
            "wandb_entity", "wandb_project", "wandb_mode", "wandb_study",
            "eval_rollout_episode", "device", "img_size", "patch_size", "embed_dim",
            "encoder_layers", "encoder_heads", "predictor_layers", "predictor_heads",
            "history_len", "dropout", "sigreg_lambda", "sigreg_projections",
            "probe_ridge", "corruption_seed", "no_amp", "extra_tag",
        })
        command = runner.train_command("python", job)
        self.assertIn("--train-data", command)
        self.assertIn("--val-data", command)
        self.assertNotIn("--encoder-checkpoint", command)
        self.assertNotIn("--train-feature-cache", command)

    def test_protocol_freezes_exact_thresholds_and_external_metric(self):
        with (
            mock.patch.object(runner, "data_snapshot", return_value={"data": {"sha256": "a"}}),
            mock.patch.object(runner, "source_snapshot", return_value={"src": {"sha256": "b"}}),
            mock.patch.object(runner, "memory_contract", return_value={"memory": "frozen"}),
        ):
            protocol = runner.build_protocol(
                "a" * 40, True,
                {"authenticated": True, "entity": runner.WANDB_ENTITY},
            )
        self.assertEqual(protocol["data_contract"]["primary_metric"], "heldout_state_nmse")
        self.assertFalse(protocol["data_contract"]["private_latent_mse_cross_model_comparison"])
        self.assertTrue(
            protocol["data_contract"]["synchronized_clean_view_targets_used_for_training"]
        )
        self.assertFalse(protocol["data_contract"]["simulator_physics_state_used_for_training"])
        final = protocol["final_success_criteria"]
        self.assertEqual(final["vs_each_ssm_and_v8"], ">=5%, >=15/25 cells, >=4/5 environments")
        self.assertEqual(final["vs_each_additive_and_scaled"], ">=2%, >=14/25, >=3/5")
        self.assertEqual(final["vs_noaction"], ">=5%, >=17/25, >=3/5")
        self.assertEqual(final["vs_static"], ">=1%, >=14/25, >=3/5")

    def test_orbit_state_validation_rejects_foreign_memory(self):
        job = next(job for job in runner.ALL_JOBS if job.design == "orbitv10")
        valid = {
            "encoder.weight": torch.zeros(1),
            "mem_orbitv10.W_o.weight": torch.zeros(1),
        }
        runner.validate_model_state(valid, job)
        invalid = dict(valid)
        invalid["mem_hacssmv8.W_o.weight"] = torch.zeros(1)
        with self.assertRaises(runner.RunnerError):
            runner.validate_model_state(invalid, job)

    def test_pilot_decision_fails_closed_on_label_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pilot_decision.json"
            runner.shared.atomic_write_json(path, {
                "pilot_screen_passed": False,
                "decision": "PILOT_CONFIRMATION_PASS",
                "scope": "prospectively_heldout_end_to_end_confirmation",
            })
            with mock.patch.object(runner, "PILOT_DECISION_PATH", path):
                with self.assertRaises(runner.RunnerError):
                    runner.read_pilot_decision()


if __name__ == "__main__":
    unittest.main()
