#!/usr/bin/env python3
"""Focused deterministic-data and tiny end-to-end trainer tests for V10."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.hacssm_v10_data import (
    VIEWS,
    V10TrajectoryDataset,
    content_sha256,
    load_cache,
    sidecar_path,
    write_cache,
)
from scripts.train_hacssm_v10 import (
    DESIGNS,
    build_model,
    encode_joint_clean_deterministic,
    main as train_main,
)


def _arrays(episodes: int, length: int, size: int, action_dim: int,
            state_dim: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    obs = rng.integers(0, 256, (episodes, length, size, size, 3), dtype=np.uint8)
    actions = rng.uniform(-1, 1, (episodes, length - 1, action_dim)).astype(np.float32)
    states = rng.normal(size=(episodes, length, state_dim)).astype(np.float64)
    return {
        "obs": obs,
        "actions": actions,
        "physics_state": states,
        "rewards": rng.normal(size=(episodes, length - 1)).astype(np.float32),
        "action_min": np.full(action_dim, -1, dtype=np.float32),
        "action_max": np.full(action_dim, 1, dtype=np.float32),
    }


def _cache(root: Path, split: str, seed: int) -> Path:
    path = root / f"tiny_{split}.npz"
    write_cache(
        path,
        env_id="walker.walk",
        split=split,
        seed=seed,
        length=24,
        img_size=16,
        smooth_rho=0.85,
        arrays=_arrays(4, 24, 16, 2, 5, seed),
    )
    return path


def test_cache_and_corruptions_are_deterministic() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = _cache(Path(directory), "train", 1)
        metadata = load_cache(path)
        assert metadata.episodes == 4 and metadata.length == 24
        assert metadata.action_dim == 2 and metadata.state_dim == 5
        assert sidecar_path(path).is_file()
        with np.load(path, allow_pickle=False) as source:
            assert str(source["content_sha256"]) == content_sha256(
                {name: source[name] for name in source.files})

        clean = V10TrajectoryDataset(path, "clean", corruption_seed=10012)
        for view in VIEWS:
            left = V10TrajectoryDataset(path, view, corruption_seed=10012)[0]
            right = V10TrajectoryDataset(path, view, corruption_seed=10012)[0]
            assert torch.equal(left["observed"], right["observed"])
            assert torch.equal(left["clean"], right["clean"])
            assert int(left["gap_start"]) < int(left["gap_end"])
            if view == "clean":
                assert torch.equal(left["observed"], clean[0]["clean"])
                assert not bool(left["corruption_mask"].any())
            else:
                assert bool(left["corruption_mask"].any())
                assert not torch.equal(left["observed"], left["clean"])


def test_all_designs_build_with_causal_normalization() -> None:
    class Args:
        img_size = 16
        patch_size = 4
        embed_dim = 8
        encoder_layers = 1
        encoder_heads = 2
        predictor_layers = 1
        predictor_heads = 2
        history_len = 3
        dropout = 0.0
        sigreg_lambda = 0.1
        sigreg_projections = 4

    observations = torch.rand(2, 8, 3, 16, 16)
    actions = torch.rand(2, 7, 2)
    for design in DESIGNS:
        args = Args()
        args.memory_mode = design
        model = build_model(args, action_dim=2)
        joint_clean = model.encode(observations)
        output = model.compute_loss(
            observations, actions, target_embeddings=joint_clean,
            diversity_embeddings=joint_clean, objective="v10j",
            detach_target_embeddings=False)
        assert torch.isfinite(output["loss"])
        assert model.encoder_norm == "causal" and model.predictor_norm == "none"


def test_clean_student_diversity_forward_is_deterministic() -> None:
    class Args:
        img_size = 16
        patch_size = 4
        embed_dim = 8
        encoder_layers = 1
        encoder_heads = 2
        predictor_layers = 1
        predictor_heads = 2
        history_len = 3
        dropout = 0.5
        sigreg_lambda = 0.1
        sigreg_projections = 4
        memory_mode = "orbitv10"

    model = build_model(Args(), action_dim=2)
    model.train()
    frame = torch.rand(1, 1, 3, 16, 16)
    duplicates = frame.expand(3, 5, -1, -1, -1).clone()
    first = encode_joint_clean_deterministic(model, duplicates)
    second = encode_joint_clean_deterministic(model, duplicates)
    assert model.encoder.training is True
    assert torch.equal(first, second)
    reference = first[:, :1].expand_as(first)
    assert torch.equal(first, reference)
    first.sum().backward()
    assert model.encoder.patch_embed.weight.grad is not None
    assert torch.isfinite(model.encoder.patch_embed.weight.grad).all()


def test_tiny_cli_smoke_writes_complete_local_payload() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        train = _cache(root, "train", 11)
        val = _cache(root, "val", 12)
        output = root / "runs"
        train_main([
            "--train-data", str(train),
            "--val-data", str(val),
            "--memory-mode", "orbitv10",
            "--seed", "7",
            "--output-dir", str(output),
            "--epochs", "1",
            "--batch-size", "2",
            "--num-workers", "0",
            "--img-size", "16",
            "--patch-size", "4",
            "--embed-dim", "8",
            "--encoder-layers", "1",
            "--encoder-heads", "2",
            "--predictor-layers", "1",
            "--predictor-heads", "2",
            "--dropout", "0",
            "--sigreg-projections", "4",
            "--no-amp",
            "--no-wandb",
            "--device", "cpu",
        ])
        run = output / "lewm-dmc:walker.walk-orbitv10-s7"
        assert (run / "model.pt").is_file()
        assert (run / "metrics.json").is_file()
        assert (run / "eval_rollout.npz").is_file()
        metrics = json.loads((run / "metrics.json").read_text())
        assert np.isfinite(metrics["heldout_state_nmse"])
        assert metrics["encoder_norm"] == "causal"
        assert metrics["predictor_norm"] == "none"
        assert metrics["ema_target_active"] is False
        assert metrics["target_stop_gradient"] is False
        assert metrics["clean_target_gradient_active"] is True
        assert metrics["training_objective"] == (
            "v10j_joint_pred_variance_covariance_equal_weight")
        assert metrics["vicreg_gradient_active"] is True
        assert metrics["sigreg_gradient_active"] is False
        assert metrics["prediction_loss_weight"] == 1.0
        assert metrics["variance_loss_weight"] == 1.0
        assert metrics["covariance_loss_weight"] == 1.0
        assert metrics["final_train_loss"] > 0.0
        assert metrics["val_pred_loss"] > 0.0
        assert metrics["encoder_mean_channel_variance"] > 1e-5
        assert metrics["encoder_covariance_effective_rank"] > 2.0
        assert metrics["encoder_singleton_max_abs"] <= 1e-6
        assert metrics["encoder_prefix_max_abs"] <= 1e-6
        checkpoint = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
        assert set(checkpoint) == {
            "model_state_dict", "args", "final_metrics", "history", "state_probe"}
        assert checkpoint["final_metrics"] == metrics
        assert len(checkpoint["history"]) == 1
        with np.load(run / "eval_rollout.npz", allow_pickle=False) as rollout:
            required = {
                "schema_version", "episode_index", "target_times", "condition",
                "phase", "state_target", "state_prediction", "state_nmse",
            }
            assert required.issubset(rollout.files)
            assert set(rollout["condition"].astype(str)) == {
                "freeze", "gaussian_noise", "checkerboard", "long_freeze"}


def main() -> None:
    test_cache_and_corruptions_are_deterministic()
    test_all_designs_build_with_causal_normalization()
    test_clean_student_diversity_forward_is_deterministic()
    test_tiny_cli_smoke_writes_complete_local_payload()
    print("V10 data/trainer tests passed.")


if __name__ == "__main__":
    main()
