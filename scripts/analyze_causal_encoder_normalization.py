#!/usr/bin/env python3
"""Audit future-time mixing from the ViT projector BatchNorm.

This is a no-retraining sensitivity analysis over the 36 checkpoints used in the
synthetic memory comparison.  It contrasts the repository's canonical encoder call,
which flattens all ``B * L`` frames before BatchNorm, with a time-slice call that
encodes all ``B`` episodes independently at each time ``t``.  The latter removes
mixing across time, but intentionally retains cross-episode (transductive) BatchNorm.

The selected checkpoint hashes, configurations, and published-metric parity are
validated before any result is committed.  Five scientific CSVs are written with
the manifest written last; every individual file is replaced atomically.

Commands::

    # Controlled 4,000-episode cohort (default)
    .venv/bin/python scripts/analyze_causal_encoder_normalization.py

    # Reproduce the original mixed-budget audit without overwriting its archive
    .venv/bin/python scripts/analyze_causal_encoder_normalization.py \
        --none-multi-root outputs/4ens --out /tmp/causal_encoder_audit_4ens
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import sklearn
import torch
import torch.nn.functional as F


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lewm.data import generate_eval_batch  # noqa: E402
from lewm.eval.memory_probe import _fit_probe  # noqa: E402
from lewm.models.memory_model import MemoryLeWorldModel  # noqa: E402


ENVS = ("tmaze", "distractor", "recall", "occlusion")
DESIGNS = ("none", "multi", "smt")
SEEDS = (0, 1, 2)
EVAL_N = 256
EVAL_SEED = 4242
SPLIT_SEED = 0
PROBE_TRAIN_FRACTION = 0.7
GLOBAL_SEED = 0

# The audit is tied to these immutable checkpoint bytes.  Refusing substituted
# checkpoints is important because master-metric parity alone does not identify a
# checkpoint uniquely.
EXPECTED_CHECKPOINT_SHA256 = {
    "lewm-tmaze-none-s0": "88d05e996f1bb26c213297c6317cc828a4278a7bda2153f5a39253a8f04605ba",
    "lewm-tmaze-none-s1": "987f2f2cf2e8a1c2310df62a40ed0c76cbab58533ee7f20244d13f65668b80e0",
    "lewm-tmaze-none-s2": "96a957e8b337877a8ed4a880c7d1314887b3aec31a8bc2be69e51732e4cf417f",
    "lewm-tmaze-multi-s0": "cd239cd6605d37cb5a4ab393bbd4faca294229f2a8750587b4fcaa619a1fd645",
    "lewm-tmaze-multi-s1": "08dacf5a82c3e6c19d2ef037c851d9403189a5d99af0516e88d0e02a918eb7cd",
    "lewm-tmaze-multi-s2": "34900abc4b0895a9f6d3fb76f2b020e57e52005e82913482ab7937b9e9af75d3",
    "lewm-tmaze-smt-s0": "9399e14f701c181130e8e8743a50fdfd7ebfbd6d1c6e35d2079103a8af269c8a",
    "lewm-tmaze-smt-s1": "3e5606c0614924eb4149af763352460a018560297f0eb41ba417da8adb2b6d2e",
    "lewm-tmaze-smt-s2": "463107f3020335dc2dcda9d59de0a49141dce3661b9bbe3392d1bd11516462c8",
    "lewm-distractor-none-s0": "0930a102e57c61d0ad39d72974768138f579e190910ca0a584d25bef225a951e",
    "lewm-distractor-none-s1": "838337baa3ace16505a3287013183afa60c91b813ebbe2ff2a57b584d12c8751",
    "lewm-distractor-none-s2": "cfaae07599d3f94093ca937576aaca6da4ba72fb9fc00ee7ec6a265a8a89c2d6",
    "lewm-distractor-multi-s0": "a31de00ab5cadfbd531fd5c3a13de36e3b3b7ece40499850f6f09f2f4dfa91f0",
    "lewm-distractor-multi-s1": "187ba9e4f07a48c8cea55d8617dad06cb4cde8014148e55ca0cbcf61f2e44fc0",
    "lewm-distractor-multi-s2": "fdf84d33cd7f42525406fa6989d4e9e19acae7d76f058d3483bb1ad79ebba747",
    "lewm-distractor-smt-s0": "c9f7ae42f0c04f3a9fafea9c6ea7935b60a3d6e0919508f038489515c83ddb4a",
    "lewm-distractor-smt-s1": "1579f01539bc89b0b22fa171ffb335646158befd027d3ee8101be215f49fe1b8",
    "lewm-distractor-smt-s2": "0c7f8af699d541c4f7a5edaffc9e72c376b49518627e54b3dc962ef44e315def",
    "lewm-recall-none-s0": "60a3adedd136fa4a22c9b8f1720b008dd98fdca766a2be16bcbed26e7d6a6343",
    "lewm-recall-none-s1": "26adb78c764f8cab60103b18713d9806fd8102bdcd974974e5cf1b9ae20b2be3",
    "lewm-recall-none-s2": "38e02c1a13eb9a33846a077be52cac3d1d87ee1cf6310cd55698173e963469f6",
    "lewm-recall-multi-s0": "22f645cd7036cf2bb5a8f2a29467547b6c380deaf3132e89ba6f00077cbd3caf",
    "lewm-recall-multi-s1": "45fa8493348f7edb3351a03b151d8f839ce7cf2677646c76925e5ebcb9f7a6d1",
    "lewm-recall-multi-s2": "20371aa8208799f0624253928d5407e33a3d42dc805991dccef8b23f972f2921",
    "lewm-recall-smt-s0": "92e109976c42179a75b815352e71f93ddb0e65373f03474dc81f655544cae766",
    "lewm-recall-smt-s1": "404be3ae87773ec97511ede39472f3ad60d6f16a9386055c09cb1ada3f0824e8",
    "lewm-recall-smt-s2": "090170057fb0d1aabafd1283876a72c0ab0fe302c6ad9ff31b6bfa1343b308dd",
    "lewm-occlusion-none-s0": "bb4ff8e6d178a7059954f2b656bc70b41f4247c88cc7da88d3a37ddfd98b8b54",
    "lewm-occlusion-none-s1": "ea202609dd0c220b199f013ff6a200eae1fbd97ee5fc703e81b45c83c362cd2e",
    "lewm-occlusion-none-s2": "b8dd5f96c1b1877baad9fb69060ae5d84a35aa8662fbb7e60444a4cd2deb1fed",
    "lewm-occlusion-multi-s0": "5896846032cb8735fc5d8f9a1c65e92d91fa215681065744fd2e299fa0fadabb",
    "lewm-occlusion-multi-s1": "e4c7fc4c77b62dc7b16f458635f1b3e2e88be9b641040bf4ab560cccdf1c6988",
    "lewm-occlusion-multi-s2": "4fe4127e146a9657373b846684209912cb0374158b5c708435ac5290a2f3b10f",
    "lewm-occlusion-smt-s0": "4cf2e2b360fd7c5f80a301603157493d66601ba0a2106f5c850a69e356ed5625",
    "lewm-occlusion-smt-s1": "b8092237dbf538c5fbd0745a1f8ce277e7f0f37b8eec9cffeded19733765347f",
    "lewm-occlusion-smt-s2": "32a103b6392b5a075eb5776d1e7376dceb0e36c6413edc59762ebf94bcc22e77",
}

# Training-matched 4,000-episode none/multi checkpoints.  The original audit used
# outputs/4ens (5,000 episodes) for these two designs; outputs/smt is the controlled
# cohort trained by the same runner/budget as outputs/smt_v2.
MATCHED_NONE_MULTI_SHA256 = {
    "lewm-tmaze-none-s0": "4c00274387936a17325ae0aa50ba15f14235e13948ab91ca3cea795848459527",
    "lewm-tmaze-none-s1": "36531f15f93a9b216f11860568125da77213f358e5cb68656d2bb375d64198d7",
    "lewm-tmaze-none-s2": "37f266482a54c8ef42581c47ade77d2e13a113ea575560223ae7491b9bf1a908",
    "lewm-tmaze-multi-s0": "10f66f08f9cdf6e622c88947d05d6e752c43e0e743f499967742c6d13d05a01a",
    "lewm-tmaze-multi-s1": "c77e0e4e7b6496ef625d7da777ce1a8a9af5f80adb59489c7773844d7f49d104",
    "lewm-tmaze-multi-s2": "b8be60b10271d0b556f48e75f9b7f07f363fdf7c1635a818ff523111369c3a6d",
    "lewm-distractor-none-s0": "438f5c00b4c3dcf55f9cef3c7a11a02aa2116a1f3c4838c5ae8c7e11c9bd28bd",
    "lewm-distractor-none-s1": "8c4cf00462796692533436d8704b545767967b0166238201255ec80e93cf7808",
    "lewm-distractor-none-s2": "5b0cbddb41f57b19ab2c087444906f4449c973e6a6f684cfaee74e7a0845d002",
    "lewm-distractor-multi-s0": "d68cdfaa4d75d5385d73125bdd4128243e3051ffe25ddb2b11446d6503ded5d1",
    "lewm-distractor-multi-s1": "8ca2df61338216cec6e0c30c6d051430a3f19ce6f047cd8cd89c4a83ef335434",
    "lewm-distractor-multi-s2": "1e6fdbb9170cd66f70956baf336e320eafe7922711c206b4cfcc52f23220f34a",
    "lewm-recall-none-s0": "0efca85fc0772aae69d2ecb3e69996b1d9ad07aaa54ff2bc3c13a643218bd125",
    "lewm-recall-none-s1": "dcf9c4fa7a3770ca2583813ff30a6430a45b2a0fc62c938e76f705ecfc69de0d",
    "lewm-recall-none-s2": "42afa5f8b03ae578cef53cf4c35a5c00082edb73881d820c4cf092d33f7cca7c",
    "lewm-recall-multi-s0": "f6275db8f72327e76917655047dfabbc1e97a90375ed40f44c3957369c894062",
    "lewm-recall-multi-s1": "33bac4986d037abc8bdb4b76fba6f1484679ff4a0cc49c3a59e4df518f84039e",
    "lewm-recall-multi-s2": "a99d77de50788c5b9c12e8913f7386b615983684fc675069a324b4289ecb868a",
    "lewm-occlusion-none-s0": "8303f862ef35a14422abd576c25ef3f10db076ceaa3a8e45158f70228b7e3f98",
    "lewm-occlusion-none-s1": "25a75a0400a58eea25f3d2982227902526f5435c1ec5499f94185555744c544f",
    "lewm-occlusion-none-s2": "7b1e64fa5f8ec2017c3abc0cfb2e088b8c8bf178f04fac97a26f06662e417821",
    "lewm-occlusion-multi-s0": "6acb773700454e711019bf5301f0b0c4cdf8103efc1cb84c4a7900e99e0e6cdf",
    "lewm-occlusion-multi-s1": "673188658eef073a392323e7b385d56f2c91050e87f80b6769eecc10824b0c14",
    "lewm-occlusion-multi-s2": "482fa60832cdb1c51965f7ae7fe64210f5ae1e74f750b2df3a892f2ec7bd0496",
}

PER_RUN_FIELDS = (
    "run", "source_root", "checkpoint", "checkpoint_sha256", "env", "design", "seed",
    "eval_n", "eval_seed", "split_seed", "length", "history_len", "reveal", "cue_end",
    "n_classes", "chance", "canonical_master_usage", "canonical_recomputed_usage",
    "causal_timeslice_usage", "usage_delta_causal_minus_canonical",
    "canonical_master_val_mse", "canonical_recomputed_published_val_mse",
    "canonical_full_self_val_mse", "causal_timeslice_self_val_mse",
    "self_val_mse_delta_causal_minus_canonical", "latent_mse_causal_vs_canonical",
    "latent_mae_causal_vs_canonical", "latent_max_abs_causal_vs_canonical",
    "latent_cosine_causal_vs_canonical", "reveal_history_latent_mse",
    "prediction_reveal_mse_causal_vs_canonical",
    "prediction_reveal_cosine_causal_vs_canonical", "pre_bn_mse_causal_vs_canonical",
    "pre_bn_max_abs_causal_vs_canonical", "pre_bn_global_mean_channel_variance",
    "pre_bn_within_time_mean_channel_variance", "pre_bn_between_time_mean_variance",
    "pre_bn_effective_rank_entropy", "pre_bn_effective_rank_participation",
    "pre_bn_rank_rel1e8",
)

PER_TIME_FIELDS = (
    "run", "env", "design", "seed", "t", "is_reveal_history", "is_cue_visible",
    "latent_mse_causal_vs_canonical", "latent_mae_causal_vs_canonical",
    "latent_cosine_causal_vs_canonical", "pre_bn_mse_causal_vs_canonical",
    "canonical_across_episode_mean_rms", "causal_timeslice_across_episode_mean_rms",
    "canonical_across_episode_mean_std", "causal_timeslice_across_episode_mean_std",
)

CONTRAST_FIELDS = (
    "env", "comparison", "seed", "canonical_usage_gap", "causal_timeslice_usage_gap",
    "gap_delta_causal_minus_canonical",
)

GROUP_METRICS = (
    "canonical_recomputed_usage", "causal_timeslice_usage",
    "usage_delta_causal_minus_canonical", "canonical_full_self_val_mse",
    "causal_timeslice_self_val_mse", "self_val_mse_delta_causal_minus_canonical",
    "latent_mse_causal_vs_canonical", "reveal_history_latent_mse",
    "prediction_reveal_mse_causal_vs_canonical", "pre_bn_effective_rank_entropy",
)

GROUP_USAGE_FIELDS = (
    "scope", "env", "design", "n",
    *(name + suffix for name in GROUP_METRICS for suffix in ("_mean", "_sd")),
)

GROUP_CONTRAST_METRICS = (
    "canonical_usage_gap", "causal_timeslice_usage_gap",
    "gap_delta_causal_minus_canonical",
)

GROUP_CONTRAST_FIELDS = (
    "scope", "env", "comparison", "n",
    *(name + suffix for name in GROUP_CONTRAST_METRICS for suffix in ("_mean", "_sd")),
)

SOURCE_PATHS = (
    "lewm/data.py", "lewm/eval/memory_probe.py", "lewm/models/encoder.py",
    "lewm/models/leworldmodel.py", "lewm/models/memory.py",
    "lewm/models/memory_model.py", "scripts/analyze_runs.py",
    "scripts/analyze_causal_encoder_normalization.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def csv_bytes(rows: list[dict[str, Any]], fields: Iterable[str]) -> bytes:
    # The archived audit uses the csv module's RFC-style CRLF line endings.
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO / path


def display_path(path: Path) -> str:
    absolute = repo_path(path).resolve()
    try:
        return absolute.relative_to(REPO).as_posix()
    except ValueError:
        return absolute.as_posix()


def read_master(root: Path) -> dict[str, dict[str, str]]:
    path = repo_path(root) / "master_metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing canonical master metrics: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_run: dict[str, dict[str, str]] = {}
    for row in rows:
        run = row["run"]
        if run in by_run:
            raise ValueError(f"duplicate run {run!r} in {path}")
        by_run[run] = row
    return by_run


def value_or(args: dict[str, Any], name: str, default: Any) -> Any:
    value = args.get(name, default)
    return default if value is None else value


def validate_config(
    args: dict[str, Any], env: str, design: str, seed: int, root_display: str,
) -> None:
    expected_common = {
        "env": env, "memory_mode": design, "seed": seed, "img_size": 64,
        "patch_size": 8, "embed_dim": 128, "encoder_layers": 6,
        "encoder_heads": 4, "predictor_layers": 4, "predictor_heads": 8,
        "history_len": 3, "dropout": 0.1, "sigreg_lambda": 0.1,
        "sigreg_projections": 512, "length": 32,
    }
    for key, expected in expected_common.items():
        if args.get(key) != expected:
            raise ValueError(
                f"configuration mismatch for {env}/{design}/s{seed}: "
                f"{key}={args.get(key)!r}, expected {expected!r}")
    for override in ("reveal", "cue_len", "n_distract", "seq_len"):
        if args.get(override) is not None:
            raise ValueError(f"unexpected environment override {override}={args[override]!r}")
    if root_display == "outputs/4ens":
        if args.get("num_episodes") != 5000:
            raise ValueError("outputs/4ens audit requires the 5,000-episode checkpoints")
    else:
        if args.get("num_episodes") != 4000:
            raise ValueError(f"{root_display} audit requires the 4,000-episode checkpoints")
    if design == "none" and root_display == "outputs/4ens":
        expected = (value_or(args, "fixed_alpha", True), args["tau_fast"], args["tau_slow"])
        if expected != (True, 3.0, 25.0):
            raise ValueError(f"invalid none-memory configuration: {expected!r}")
    else:
        if (args["tau_fast"], args["tau_slow"]) != (2.0, 20.0):
            raise ValueError(f"invalid {design} nominal fast/slow horizons")
        if design != "none":
            taus = tuple(value_or(args, "multi_taus", ()))
            if taus != (2, 4, 8, 16, 32, 64):
                raise ValueError(f"invalid {design} timescale bank: {taus!r}")
        expected_fixed = root_display == "outputs/4ens"
        if value_or(args, "fixed_alpha", True) is not expected_fixed:
            raise ValueError(
                f"{design} fixed_alpha flag does not match cohort {root_display}")
        if design == "smt":
            if value_or(args, "encoder", "vit") != "vit":
                raise ValueError("SMT audit requires the ViT encoder")
            if value_or(args, "smt_router", "softmax") != "sigmoid":
                raise ValueError("SMT audit requires the sigmoid router")


def build_model(args: dict[str, Any]) -> MemoryLeWorldModel:
    mode = args["memory_mode"]
    impl = mode if mode in ("multi", "gru", "ssm", "retrieval", "smt", "ocsmt") else "ema"
    ema_mode = "both" if impl != "ema" else mode
    return MemoryLeWorldModel(
        img_size=args["img_size"], patch_size=args["patch_size"],
        embed_dim=args["embed_dim"], action_dim=2,
        encoder_layers=args["encoder_layers"], encoder_heads=args["encoder_heads"],
        predictor_layers=args["predictor_layers"], predictor_heads=args["predictor_heads"],
        history_len=args["history_len"], dropout=args["dropout"],
        sigreg_lambda=args["sigreg_lambda"], sigreg_projections=args["sigreg_projections"],
        memory_mode=ema_mode, tau_fast=args["tau_fast"], tau_slow=args["tau_slow"],
        learnable_alpha=not value_or(args, "fixed_alpha", True), memory_impl=impl,
        multi_taus=tuple(value_or(args, "multi_taus", (2, 4, 8, 16, 32, 64))),
        encoder_type=value_or(args, "encoder", "vit"),
        smt_router=value_or(args, "smt_router", "softmax"),
        oc_num=value_or(args, "oc_num", 28), oc_tau_min=value_or(args, "oc_tau_min", 1.5),
        oc_tau_max=value_or(args, "oc_tau_max", 256.0),
        oc_stochastic_gates=value_or(args, "oc_gate_mode", "stochastic") == "stochastic",
        l0_lambda=value_or(args, "l0_lambda", 0.0),
    )


def eval_batch(args: dict[str, Any]) -> dict[str, Any]:
    kwargs = {}
    for name in ("reveal", "cue_len", "n_distract", "seq_len"):
        if args.get(name) is not None:
            kwargs[name] = args[name]
    return generate_eval_batch(
        args["env"], EVAL_N, img_size=args["img_size"], length=args["length"],
        seed=EVAL_SEED, **kwargs)


def matched_usage(prediction: torch.Tensor, cue: np.ndarray, n_classes: int) -> float:
    permutation = np.random.default_rng(SPLIT_SEED).permutation(len(cue))
    n_train = int(PROBE_TRAIN_FRACTION * len(cue))
    train, test = permutation[:n_train], permutation[n_train:]
    array = prediction.float().cpu().numpy()
    return _fit_probe(array[train], cue[train], array[test], cue[test], n_classes)


def prediction_mse(
    model: MemoryLeWorldModel,
    z_tilde: torch.Tensor,
    actions: torch.Tensor,
    target: torch.Tensor,
) -> float:
    batch, length, dim = target.shape
    history = model.history_len
    windows = length - history
    z_windows = z_tilde.unfold(1, history, 1)[:, :windows]
    z_windows = z_windows.permute(0, 1, 3, 2).reshape(batch * windows, history, dim)
    action_windows = actions.unfold(1, history, 1)[:, :windows]
    action_windows = action_windows.permute(0, 1, 3, 2).reshape(
        batch * windows, history, model.action_dim)
    prediction = model.predictor(z_windows, action_windows)[:, -1, :]
    return float(((prediction - target[:, history:length].reshape(-1, dim)) ** 2).mean())


def effective_ranks(pre_bn: torch.Tensor) -> tuple[float, float, int]:
    flat = pre_bn.float().cpu().numpy().reshape(-1, pre_bn.shape[-1])
    eigenvalues = np.linalg.eigvalsh(np.cov(flat, rowvar=False))
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = float(eigenvalues.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("pre-BatchNorm covariance has non-positive or non-finite trace")
    probability = eigenvalues / total
    positive = probability > 0
    entropy_rank = float(np.exp(-(probability[positive] * np.log(probability[positive])).sum()))
    participation_rank = float(total ** 2 / np.square(eigenvalues).sum())
    numerical_rank = int((eigenvalues > eigenvalues.max() * 1e-8).sum())
    return entropy_rank, participation_rank, numerical_rank


def expected_checkpoint_hash(root_display: str, run: str, design: str) -> str:
    if design == "smt" and root_display == "outputs/smt_v2":
        return EXPECTED_CHECKPOINT_SHA256[run]
    if design in ("none", "multi") and root_display == "outputs/4ens":
        return EXPECTED_CHECKPOINT_SHA256[run]
    if design in ("none", "multi") and root_display == "outputs/smt":
        return MATCHED_NONE_MULTI_SHA256[run]
    raise ValueError(
        f"unsupported/unpinned checkpoint root {root_display!r} for design {design!r}; "
        "supported roots are outputs/smt (matched), outputs/4ens (archived), and outputs/smt_v2")


@torch.no_grad()
def analyze_checkpoint(
    env: str,
    design: str,
    seed: int,
    device: torch.device,
    root: Path,
    master_rows: dict[str, dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]], MemoryLeWorldModel, torch.Tensor]:
    run = f"lewm-{env}-{design}-s{seed}"
    root_display = display_path(root)
    source_root = repo_path(root).name
    checkpoint = repo_path(root) / run / "model.pt"
    checkpoint_display = f"{root_display}/{run}/model.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing selected checkpoint: {checkpoint}")
    checkpoint_hash = sha256_file(checkpoint)
    expected_hash = expected_checkpoint_hash(root_display, run, design)
    if checkpoint_hash != expected_hash:
        raise ValueError(
            f"checkpoint hash mismatch for {run}: {checkpoint_hash} != "
            f"{expected_hash}")

    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    if set(("args", "model_state_dict")) - set(saved):
        raise ValueError(f"{checkpoint}: missing args or model_state_dict")
    args = saved["args"]
    validate_config(args, env, design, seed, root_display)
    model = build_model(args).to(device)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.eval()

    master = master_rows.get(run)
    if master is None:
        raise ValueError(f"{run} is absent from {root_display}/master_metrics.csv")
    if (master["env"], master["design"], int(master["seed"])) != (env, design, seed):
        raise ValueError(f"master-metric identity mismatch for {run}")
    master_usage = float(master["usage_matched"])
    master_val_mse = float(master["val_mse"])

    batch = eval_batch(args)
    obs = batch["obs"].to(device)
    actions = batch["actions"].to(device)
    cue = batch["cue"].numpy()
    reveal_values = batch["reveal"].numpy()
    cue_end_values = batch["cue_end"].numpy()
    if not np.all(reveal_values == reveal_values[0]) or not np.all(cue_end_values == cue_end_values[0]):
        raise ValueError(f"non-constant reveal/cue_end metadata for {run}")
    reveal = int(reveal_values[0])
    cue_end = int(cue_end_values[0])
    n_classes = int(batch["n_cue_classes"])
    length = int(args["length"])
    history = int(args["history_len"])
    if obs.shape[:2] != (EVAL_N, length) or actions.shape[:2] != (EVAL_N, length - 1):
        raise ValueError(f"unexpected evaluation batch shapes for {run}")
    if n_classes < 2 or not history <= reveal < length:
        raise ValueError(f"invalid probe metadata for {run}")

    captured: list[torch.Tensor] = []

    def capture_pre_bn(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        if len(inputs) != 1:
            raise RuntimeError("unexpected BatchNorm pre-hook inputs")
        captured.append(inputs[0].detach().clone())

    hook = model.encoder.projector[1].register_forward_pre_hook(capture_pre_bn)
    try:
        z_canonical, _, _, z_tilde_canonical = model.encode_with_memory(obs)
        if len(captured) != 1:
            raise RuntimeError(f"canonical encoding triggered {len(captured)} BatchNorm calls")
        pre_bn_canonical = captured.pop().reshape(EVAL_N, length, -1)

        causal_z, causal_pre_bn = [], []
        for t in range(length):
            causal_z.append(model.encode(obs[:, t]))
            if len(captured) != 1:
                raise RuntimeError(f"time {t} encoding triggered {len(captured)} BatchNorm calls")
            causal_pre_bn.append(captured.pop())
        z_causal = torch.stack(causal_z, dim=1)
        pre_bn_causal = torch.stack(causal_pre_bn, dim=1)
        z_tilde_causal = model._inject(z_causal)
    finally:
        hook.remove()

    canonical_targets = []
    for start in range(0, EVAL_N, 64):
        canonical_targets.append(model.encode(obs[start:start + 64]))
    canonical_chunk64_target = torch.cat(canonical_targets, dim=0)

    published_val_mse = prediction_mse(
        model, z_tilde_canonical, actions, canonical_chunk64_target)
    canonical_self_val_mse = prediction_mse(model, z_tilde_canonical, actions, z_canonical)
    causal_self_val_mse = prediction_mse(model, z_tilde_causal, actions, z_causal)

    reveal_slice = slice(reveal - history, reveal)
    canonical_prediction = model.predictor(
        z_tilde_canonical[:, reveal_slice], actions[:, reveal_slice])[:, -1, :]
    causal_prediction = model.predictor(
        z_tilde_causal[:, reveal_slice], actions[:, reveal_slice])[:, -1, :]
    canonical_usage = matched_usage(canonical_prediction, cue, n_classes)
    causal_usage = matched_usage(causal_prediction, cue, n_classes)

    entropy_rank, participation_rank, numerical_rank = effective_ranks(pre_bn_canonical)
    per_run = {
        "run": run,
        "source_root": source_root,
        "checkpoint": checkpoint_display,
        "checkpoint_sha256": checkpoint_hash,
        "env": env,
        "design": design,
        "seed": seed,
        "eval_n": EVAL_N,
        "eval_seed": EVAL_SEED,
        "split_seed": SPLIT_SEED,
        "length": length,
        "history_len": history,
        "reveal": reveal,
        "cue_end": cue_end,
        "n_classes": n_classes,
        "chance": 1.0 / n_classes,
        "canonical_master_usage": master_usage,
        "canonical_recomputed_usage": canonical_usage,
        "causal_timeslice_usage": causal_usage,
        "usage_delta_causal_minus_canonical": causal_usage - canonical_usage,
        "canonical_master_val_mse": master_val_mse,
        "canonical_recomputed_published_val_mse": published_val_mse,
        "canonical_full_self_val_mse": canonical_self_val_mse,
        "causal_timeslice_self_val_mse": causal_self_val_mse,
        "self_val_mse_delta_causal_minus_canonical": causal_self_val_mse - canonical_self_val_mse,
        "latent_mse_causal_vs_canonical": float(F.mse_loss(z_causal, z_canonical)),
        "latent_mae_causal_vs_canonical": float(F.l1_loss(z_causal, z_canonical)),
        "latent_max_abs_causal_vs_canonical": float((z_causal - z_canonical).abs().max()),
        "latent_cosine_causal_vs_canonical": float(
            F.cosine_similarity(z_causal, z_canonical, dim=-1).mean()),
        "reveal_history_latent_mse": float(F.mse_loss(
            z_causal[:, reveal_slice], z_canonical[:, reveal_slice])),
        "prediction_reveal_mse_causal_vs_canonical": float(F.mse_loss(
            causal_prediction, canonical_prediction)),
        "prediction_reveal_cosine_causal_vs_canonical": float(F.cosine_similarity(
            causal_prediction, canonical_prediction, dim=-1).mean()),
        "pre_bn_mse_causal_vs_canonical": float(F.mse_loss(pre_bn_causal, pre_bn_canonical)),
        "pre_bn_max_abs_causal_vs_canonical": float(
            (pre_bn_causal - pre_bn_canonical).abs().max()),
        "pre_bn_global_mean_channel_variance": float(
            pre_bn_canonical.var(dim=(0, 1), unbiased=False).mean()),
        "pre_bn_within_time_mean_channel_variance": float(
            pre_bn_canonical.var(dim=0, unbiased=False).mean()),
        "pre_bn_between_time_mean_variance": float(
            pre_bn_canonical.mean(dim=0).var(dim=0, unbiased=False).mean()),
        "pre_bn_effective_rank_entropy": entropy_rank,
        "pre_bn_effective_rank_participation": participation_rank,
        "pre_bn_rank_rel1e8": numerical_rank,
    }

    per_time = []
    for t in range(length):
        canonical_at_t = z_canonical[:, t]
        causal_at_t = z_causal[:, t]
        canonical_episode_mean = canonical_at_t.mean(dim=0)
        causal_episode_mean = causal_at_t.mean(dim=0)
        per_time.append({
            "run": run,
            "env": env,
            "design": design,
            "seed": seed,
            "t": t,
            "is_reveal_history": int(reveal - history <= t < reveal),
            "is_cue_visible": int(t < cue_end),
            "latent_mse_causal_vs_canonical": float(F.mse_loss(causal_at_t, canonical_at_t)),
            "latent_mae_causal_vs_canonical": float(F.l1_loss(causal_at_t, canonical_at_t)),
            "latent_cosine_causal_vs_canonical": float(F.cosine_similarity(
                causal_at_t, canonical_at_t, dim=-1).mean()),
            "pre_bn_mse_causal_vs_canonical": float(F.mse_loss(
                pre_bn_causal[:, t], pre_bn_canonical[:, t])),
            "canonical_across_episode_mean_rms": float(torch.sqrt(
                canonical_episode_mean.square().mean())),
            "causal_timeslice_across_episode_mean_rms": float(torch.sqrt(
                causal_episode_mean.square().mean())),
            "canonical_across_episode_mean_std": float(
                canonical_at_t.std(dim=0, unbiased=False).mean()),
            "causal_timeslice_across_episode_mean_std": float(
                causal_at_t.std(dim=0, unbiased=False).mean()),
        })

    usage_error = abs(canonical_usage - master_usage)
    val_mse_error = abs(published_val_mse - master_val_mse)
    if usage_error != 0.0 or val_mse_error > 1e-7:
        raise ValueError(
            f"published-metric parity failure for {run}: usage error={usage_error}, "
            f"val_mse error={val_mse_error}")
    return per_run, per_time, model, obs


def mean_sd(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        raise ValueError("at least two values are required for a sample standard deviation")
    # statistics uses compensated exact-ratio summation; this is also how the archived
    # grouped tables were produced (np.mean differs by a few final decimal bits).
    return statistics.mean(values), statistics.stdev(values)


def grouped_rows(
    rows: list[dict[str, Any]],
    group_names: tuple[str, ...],
    metrics: tuple[str, ...],
    scope: str,
    env: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {"scope": scope, "env": env}
    for name in group_names:
        unique = {row[name] for row in rows}
        if len(unique) != 1:
            raise ValueError(f"group has multiple {name} values: {sorted(unique)}")
        result[name] = next(iter(unique))
    result["n"] = len(rows)
    for metric in metrics:
        result[f"{metric}_mean"], result[f"{metric}_sd"] = mean_sd(
            [float(row[metric]) for row in rows])
    return result


def aggregate(per_run: list[dict[str, Any]]) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
    by_key = {(row["env"], row["design"], row["seed"]): row for row in per_run}
    if len(by_key) != len(ENVS) * len(DESIGNS) * len(SEEDS):
        raise ValueError("per-run rows do not form the expected complete factorial")
    contrasts = []
    for env in ENVS:
        for design in ("multi", "smt"):
            comparison = f"{design}-minus-none"
            for seed in SEEDS:
                memory = by_key[env, design, seed]
                none = by_key[env, "none", seed]
                canonical_gap = (
                    memory["canonical_recomputed_usage"] - none["canonical_recomputed_usage"])
                causal_gap = memory["causal_timeslice_usage"] - none["causal_timeslice_usage"]
                contrasts.append({
                    "env": env,
                    "comparison": comparison,
                    "seed": seed,
                    "canonical_usage_gap": canonical_gap,
                    "causal_timeslice_usage_gap": causal_gap,
                    "gap_delta_causal_minus_canonical": causal_gap - canonical_gap,
                })

    grouped_usage = []
    for env in sorted(ENVS):
        for design in sorted(DESIGNS):
            selected = [r for r in per_run if r["env"] == env and r["design"] == design]
            grouped_usage.append(grouped_rows(
                selected, ("design",), GROUP_METRICS, "env", env))
    for design in sorted(DESIGNS):
        selected = [r for r in per_run if r["design"] == design]
        grouped_usage.append(grouped_rows(
            selected, ("design",), GROUP_METRICS, "overall", ""))

    grouped_contrasts = []
    comparisons = ("multi-minus-none", "smt-minus-none")
    for env in sorted(ENVS):
        for comparison in comparisons:
            selected = [
                r for r in contrasts if r["env"] == env and r["comparison"] == comparison]
            grouped_contrasts.append(grouped_rows(
                selected, ("comparison",), GROUP_CONTRAST_METRICS, "env", env))
    for comparison in comparisons:
        selected = [r for r in contrasts if r["comparison"] == comparison]
        grouped_contrasts.append(grouped_rows(
            selected, ("comparison",), GROUP_CONTRAST_METRICS, "overall", ""))
    return contrasts, grouped_usage, grouped_contrasts


def git_metadata() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, text=True,
        stdout=subprocess.PIPE).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO, check=True, text=True,
        stdout=subprocess.PIPE).stdout)
    return commit, dirty


def resolve_device(specification: str) -> torch.device:
    if specification == "auto":
        specification = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(specification)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested device {device}, but CUDA is unavailable")
    return device


def set_canonical_numerics(seed: int) -> None:
    """Pin stochastic APIs while preserving analyze_runs.py arithmetic defaults."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default="outputs/causal_encoder_audit_matched", type=Path,
        help="output directory (default: outputs/causal_encoder_audit_matched)")
    parser.add_argument(
        "--device", default="auto",
        help="PyTorch device such as cuda, cuda:0, or cpu (default: auto)")
    parser.add_argument(
        "--none-multi-root", default=Path("outputs/smt"), type=Path,
        help="root containing none/multi checkpoints (default: outputs/smt, 4k matched cohort)")
    parser.add_argument(
        "--smt-root", default=Path("outputs/smt_v2"), type=Path,
        help="root containing sigmoid-SMT checkpoints (default: outputs/smt_v2)")
    options = parser.parse_args()
    device = resolve_device(options.device)
    set_canonical_numerics(GLOBAL_SEED)
    started = time.perf_counter()

    expected_runs = {
        f"lewm-{env}-{design}-s{seed}"
        for env in ENVS for design in DESIGNS for seed in SEEDS
    }
    if set(EXPECTED_CHECKPOINT_SHA256) != expected_runs:
        raise RuntimeError("internal checkpoint-hash table does not match the selected factorial")
    matched_runs = {
        f"lewm-{env}-{design}-s{seed}"
        for env in ENVS for design in ("none", "multi") for seed in SEEDS
    }
    if set(MATCHED_NONE_MULTI_SHA256) != matched_runs:
        raise RuntimeError("internal matched checkpoint-hash table is incomplete")

    roots = {
        "none": options.none_multi_root,
        "multi": options.none_multi_root,
        "smt": options.smt_root,
    }
    masters = {
        display_path(root): read_master(root) for root in {options.none_multi_root, options.smt_root}
    }
    per_run: list[dict[str, Any]] = []
    per_time: list[dict[str, Any]] = []
    last_model = None
    last_obs = None
    for env in ENVS:
        for design in DESIGNS:
            for seed in SEEDS:
                run = f"lewm-{env}-{design}-s{seed}"
                print(f"[{len(per_run) + 1:02d}/36] {run}", flush=True)
                root = roots[design]
                row, time_rows, last_model, last_obs = analyze_checkpoint(
                    env, design, seed, device, root, masters[display_path(root)])
                per_run.append(row)
                per_time.extend(time_rows)
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    if len(per_run) != 36 or len(per_time) != 36 * 32:
        raise RuntimeError("audit produced an incomplete row set")
    contrasts, grouped_usage, grouped_contrasts = aggregate(per_run)

    assert last_model is not None and last_obs is not None
    single_frame_test: dict[str, Any]
    try:
        with torch.no_grad():
            last_model.encode(last_obs[:1, 0])
    except Exception as error:  # the exact failure is itself an audited result
        single_frame_test = {
            "exception_message": str(error),
            "exception_type": type(error).__name__,
            "supported": False,
        }
    else:
        single_frame_test = {"exception_message": "", "exception_type": "", "supported": True}

    payloads = {
        "per_run.csv": csv_bytes(per_run, PER_RUN_FIELDS),
        "per_time.csv": csv_bytes(per_time, PER_TIME_FIELDS),
        "memory_contrasts.csv": csv_bytes(contrasts, CONTRAST_FIELDS),
        "grouped_usage.csv": csv_bytes(grouped_usage, GROUP_USAGE_FIELDS),
        "grouped_contrasts.csv": csv_bytes(grouped_contrasts, GROUP_CONTRAST_FIELDS),
    }
    out = options.out
    for name, payload in payloads.items():
        atomic_write(out / name, payload)

    max_usage_parity = max(
        abs(r["canonical_recomputed_usage"] - r["canonical_master_usage"]) for r in per_run)
    max_val_mse_parity = max(
        abs(r["canonical_recomputed_published_val_mse"] - r["canonical_master_val_mse"])
        for r in per_run)
    commit, dirty = git_metadata()
    output_hashes = {
        (out / name).as_posix(): sha256_file(out / name) for name in sorted(payloads)
    }
    source_hashes = {name: sha256_file(REPO / name) for name in SOURCE_PATHS}
    runtime_device = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or "CPU")
    manifest = {
        "audit": "future-time BatchNorm leakage sensitivity audit (no retraining)",
        "canonical_parity": {
            "max_abs_published_val_mse_difference_vs_master": max_val_mse_parity,
            "max_abs_usage_difference_vs_master": max_usage_parity,
        },
        "evaluation": {
            "episodes": EVAL_N,
            "eval_seed": EVAL_SEED,
            "probe": (
                "StandardScaler + sklearn LogisticRegression(max_iter=300,C=1.0), "
                "train/test on predicted reveal latents"),
            "probe_split_seed": SPLIT_SEED,
            "probe_train_fraction": PROBE_TRAIN_FRACTION,
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_worktree_dirty": dirty,
        "output_sha256": output_hashes,
        "pre_bn_rank_definition": {
            "entropy_effective_rank": "exp(entropy of covariance eigenvalues)",
            "numerical_rank": "count eigenvalues > max_eigenvalue*1e-8",
            "participation_rank": "(sum eigenvalues)^2/sum(eigenvalues^2)",
        },
        "protocols": {
            "canonical_full": (
                "one model.encode_with_memory call on (B,L,C,H,W); encode flattens B*L "
                "before projector BatchNorm"),
            "causal_timeslice": (
                "for each t, one model.encode call on obs[:,t] (all B episodes, no other times); "
                "stack z over t, then unchanged causal memory injection and predictor"),
            "prediction_mse": (
                "self-consistent target normalization within each protocol; published canonical "
                "chunk-64 target MSE is separately reproduced for parity"),
        },
        "runtime": {
            "cuda": torch.version.cuda,
            "device": runtime_device,
            "numpy": np.__version__,
            "python": f"Python {platform.python_version()}",
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
            "wall_seconds": time.perf_counter() - started,
        },
        "numerical_backend": {
            "global_seed": GLOBAL_SEED,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "scope_warning": (
            "The time-slice condition removes future/past time mixing but remains transductive "
            "across all 256 evaluation episodes at each t. It is not independent single-episode "
            "inference and checkpoints were not retrained for this normalization context."),
        "selection": {
            "checkpoint_count": len(per_run),
            "designs": list(DESIGNS),
            "environments": list(ENVS),
            "roots": {
                "none_multi": display_path(options.none_multi_root),
                "smt": display_path(options.smt_root),
            },
            "seeds": list(SEEDS),
        },
        "single_frame_single_episode_encoder_test": single_frame_test,
        "source_sha256": source_hashes,
    }
    manifest_payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_write(out / "manifest.json", manifest_payload)
    print(f"wrote {out} ({len(per_run)} runs, {len(per_time)} run-time rows)")


if __name__ == "__main__":
    main()
