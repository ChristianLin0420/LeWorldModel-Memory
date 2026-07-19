#!/usr/bin/env python3
"""OGBench MESM admission bridge.

This script is a protocol bridge, not a frozen-host memory result.  It checks
whether an OGBench environment can support the MESM counterfactual-cue ladder:
the cue is decodable when visible, but not from endpoint visuals, actions, or
proprio/state observations after the cue disappears.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
from PIL import Image
import torch
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/ogbench_mesmem_admission_v1"
DEFAULT_DINOV2 = ROOT / "outputs/dinowm_native_pusht_audit_v1/vendor/dinov2"
DEFAULT_TORCH_HOME = ROOT / "outputs/dinowm_native_pusht_audit_v1/torch_home"

CLASSES = 4
LENGTH = 20
LAST_CUE_FRAME = 3
AGES = (4, 8, 15)
CARD_POSITIONS = np.asarray([[16, 16], [176, 16], [16, 176], [176, 176]],
                            dtype=np.int64)
CARD_COLOR = (230, 57, 70)
CARD_SIZE = 28


def stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, indent=2) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dinov2", type=Path, default=DEFAULT_DINOV2)
    parser.add_argument("--torch-home", type=Path, default=DEFAULT_TORCH_HOME)
    parser.add_argument("--train-bases", type=int, default=200)
    parser.add_argument("--validation-bases", type=int, default=80)
    parser.add_argument("--seed", type=int, default=881003)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in ("output", "dinov2", "torch_home"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    args.output = args.output / args.env_name.replace("/", "_")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resize_frame(frame: np.ndarray, size: int = 224) -> np.ndarray:
    return np.asarray(Image.fromarray(frame.astype(np.uint8)).resize(
        (size, size), Image.Resampling.BILINEAR), dtype=np.uint8)


def draw_cue(frame: np.ndarray, label: int) -> np.ndarray:
    image = Image.fromarray(frame.copy())
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    x, y = [int(v) for v in CARD_POSITIONS[int(label)]]
    draw.rectangle([x - 6, y - 6, x + CARD_SIZE + 6, y + CARD_SIZE + 6],
                   fill=(255, 255, 255), outline=(17, 24, 39), width=2)
    draw.ellipse([x, y, x + CARD_SIZE, y + CARD_SIZE],
                 fill=CARD_COLOR, outline=(17, 24, 39), width=1)
    return np.asarray(image, dtype=np.uint8)


def render_variant(frames: np.ndarray, label: int) -> np.ndarray:
    variant = frames.copy()
    for t in range(1, LAST_CUE_FRAME + 1):
        variant[t] = draw_cue(variant[t], label)
    return variant


def collect_env(args: argparse.Namespace) -> dict[str, np.ndarray]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    import ogbench  # noqa: WPS433

    total = int(args.train_bases + args.validation_bases)
    rng = np.random.default_rng(args.seed)
    env = ogbench.make_env_and_datasets(args.env_name, env_only=True)
    frames, observations, actions = [], [], []
    for episode in range(total):
        obs, _ = env.reset(seed=int(args.seed + episode))
        ep_frames = [resize_frame(env.render())]
        ep_obs = [np.asarray(obs, dtype=np.float32).reshape(-1)]
        ep_actions = []
        for _ in range(LENGTH - 1):
            action = env.action_space.sample()
            if rng.random() < 0.05:
                action = np.zeros_like(action)
            obs, _, terminated, truncated, _ = env.step(action)
            ep_actions.append(np.asarray(action, dtype=np.float32).reshape(-1))
            ep_obs.append(np.asarray(obs, dtype=np.float32).reshape(-1))
            ep_frames.append(resize_frame(env.render()))
            if terminated or truncated:
                obs, _ = env.reset(seed=int(args.seed + episode))
        frames.append(np.stack(ep_frames))
        observations.append(np.stack(ep_obs))
        actions.append(np.stack(ep_actions))
    env.close()
    return {
        "frames": np.stack(frames).astype(np.uint8),
        "observations": np.stack(observations).astype(np.float32),
        "actions": np.stack(actions).astype(np.float32),
    }


def load_dinov2(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    os.environ["TORCH_HOME"] = str(args.torch_home.resolve())
    model = torch.hub.load(str(args.dinov2.resolve()), "dinov2_vits14",
                           source="local", pretrained=True)
    return model.eval().to(device)


@torch.no_grad()
def encode_frames(model: torch.nn.Module, frames: np.ndarray,
                  device: torch.device, batch_size: int) -> np.ndarray:
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF

    outputs = []
    for start in range(0, len(frames), batch_size):
        rows = frames[start:start + batch_size]
        tensor = torch.from_numpy(rows.copy()).to(device)
        tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
        tensor = tensor.sub_(0.5).div_(0.5)
        tensor = TF.resize(tensor, [196, 196],
                           interpolation=InterpolationMode.BILINEAR,
                           antialias=True)
        outputs.append(model.forward_features(tensor)[
            "x_norm_patchtokens"].float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def pooled(features: np.ndarray) -> np.ndarray:
    grid = features.reshape(features.shape[0], 14, 14, 384)
    pieces = []
    for level in (1, 2, 4):
        for y in range(level):
            y0, y1 = y * 14 // level, (y + 1) * 14 // level
            for x in range(level):
                x0, x1 = x * 14 // level, (x + 1) * 14 // level
                pieces.append(grid[:, y0:y1, x0:x1].mean(axis=(1, 2)))
    return np.concatenate(pieces, axis=1).astype(np.float32)


def labels_for(n: int) -> np.ndarray:
    return np.tile(np.arange(CLASSES, dtype=np.int64), n)


def expand_base(x: np.ndarray) -> np.ndarray:
    return np.repeat(x, CLASSES, axis=0)


def fit_predict(train_x: np.ndarray, train_y: np.ndarray,
                val_x: np.ndarray) -> np.ndarray:
    clf = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0))
    clf.fit(train_x, train_y)
    return clf.predict(val_x).astype(np.int64)


def record(pred: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "confusion_matrix": confusion_matrix(
            truth, pred, labels=list(range(CLASSES))).astype(int).tolist(),
        "count": int(len(truth)),
    }


def split_arrays(data: dict[str, np.ndarray],
                 train_bases: int) -> tuple[dict[str, np.ndarray],
                                             dict[str, np.ndarray]]:
    return ({k: v[:train_bases] for k, v in data.items()},
            {k: v[train_bases:] for k, v in data.items()})


def build_features(args: argparse.Namespace, model: torch.nn.Module,
                   device: torch.device, data: dict[str, np.ndarray]) \
        -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cue_frames = []
    endpoints = {age: [] for age in AGES}
    outside = pairwise = post_cue = 0
    min_changed = None
    mask = np.zeros((224, 224), dtype=bool)
    for x, y in CARD_POSITIONS:
        mask[max(0, y - 6):min(224, y + CARD_SIZE + 7),
             max(0, x - 6):min(224, x + CARD_SIZE + 7)] = True
    for frames in data["frames"]:
        variants = [render_variant(frames, label) for label in range(CLASSES)]
        for variant in variants:
            cue_frames.append(variant[LAST_CUE_FRAME])
            changed = int(np.count_nonzero(
                variant[1:LAST_CUE_FRAME + 1]
                != frames[1:LAST_CUE_FRAME + 1]))
            min_changed = changed if min_changed is None else min(min_changed, changed)
            post_cue += int(np.count_nonzero(
                variant[LAST_CUE_FRAME + 1:] != frames[LAST_CUE_FRAME + 1:]))
            outside += int(np.count_nonzero(
                (variant[:LAST_CUE_FRAME + 1] != frames[:LAST_CUE_FRAME + 1])
                & ~mask[None, :, :, None]))
        for a in range(CLASSES):
            for b in range(a + 1, CLASSES):
                pairwise += int(np.count_nonzero(
                    (variants[a][:LAST_CUE_FRAME + 1]
                     != variants[b][:LAST_CUE_FRAME + 1])
                    & ~mask[None, :, :, None]))
        for age in AGES:
            endpoints[age].append(frames[LAST_CUE_FRAME + age])
    features = {
        "cue": pooled(encode_frames(
            model, np.stack(cue_frames), device, args.batch_size)),
        "endpoint": {
            age: pooled(encode_frames(
                model, np.stack(values), device, args.batch_size))
            for age, values in endpoints.items()
        },
    }
    audit = {
        "base_count": int(len(data["frames"])),
        "expanded_count": int(len(data["frames"]) * CLASSES),
        "outside_declared_mask_changed_pixels": int(outside),
        "pairwise_outside_mask_changed_pixels": int(pairwise),
        "post_cue_differing_pixels": int(post_cue),
        "minimum_changed_cue_pixels_any_label": int(min_changed or 0),
    }
    return features, audit


def main() -> None:
    args = resolve_args(parse_args())
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    data = collect_env(args)
    train, val = split_arrays(data, int(args.train_bases))
    model = load_dinov2(args, device)
    train_feat, train_audit = build_features(args, model, device, train)
    val_feat, val_audit = build_features(args, model, device, val)
    y_train = labels_for(int(args.train_bases))
    y_val = labels_for(int(args.validation_bases))
    cue = record(fit_predict(train_feat["cue"], y_train, val_feat["cue"]), y_val)
    shortcuts = {}
    for age in AGES:
        endpoint_pred = fit_predict(
            expand_base(train_feat["endpoint"][age]), y_train,
            expand_base(val_feat["endpoint"][age]))
        endpoint = LAST_CUE_FRAME + age
        action_pred = fit_predict(
            expand_base(train["actions"][:, endpoint - 1]), y_train,
            expand_base(val["actions"][:, endpoint - 1]))
        obs_pred = fit_predict(
            expand_base(train["observations"][:, endpoint - 1]), y_train,
            expand_base(val["observations"][:, endpoint - 1]))
        shortcuts[str(age)] = {
            "no_cue_visual_endpoint": record(endpoint_pred, y_val),
            "action_only": record(action_pred, y_val),
            "observation_only": record(obs_pred, y_val),
        }
    shortcut_max = 1.0 / CLASSES + 0.05
    shortcut_pass = all(
        item["balanced_accuracy"] <= shortcut_max
        for group in shortcuts.values() for item in group.values())
    counterfactual_pass = all(
        item["outside_declared_mask_changed_pixels"] == 0
        and item["pairwise_outside_mask_changed_pixels"] == 0
        and item["post_cue_differing_pixels"] == 0
        for item in (train_audit, val_audit))
    admitted = bool(cue["balanced_accuracy"] >= 0.75
                    and shortcut_pass and counterfactual_pass)
    sample = np.concatenate([
        render_variant(val["frames"][0], label)[LAST_CUE_FRAME]
        for label in range(CLASSES)
    ], axis=1)
    Image.fromarray(sample).save(args.output / "cue_variants.png")
    summary = {
        "schema": "ogbench_mesmem_admission_v1",
        "status": "admitted" if admitted else "stopped_fail_closed",
        "admitted": admitted,
        "env_name": args.env_name,
        "selection": {
            "seed": int(args.seed),
            "train_bases": int(args.train_bases),
            "validation_bases": int(args.validation_bases),
            "expanded_train": int(args.train_bases * CLASSES),
            "expanded_validation": int(args.validation_bases * CLASSES),
        },
        "cue_encoding": {**cue, "minimum": 0.75,
                         "pass": cue["balanced_accuracy"] >= 0.75},
        "shortcut_threshold": shortcut_max,
        "shortcut_pass": bool(shortcut_pass),
        "shortcuts": shortcuts,
        "counterfactual_audit": {
            "train": train_audit,
            "validation": val_audit,
            "pass": bool(counterfactual_pass),
        },
        "claim_boundary": (
            "admission bridge only; no frozen world-model carrier or "
            "executed-use claim"),
        "artifacts": {"cue_variants": "cue_variants.png"},
    }
    (args.output / "admission_summary.json").write_text(stable_json(summary))
    print(stable_json({
        "status": summary["status"],
        "env_name": args.env_name,
        "cue_bacc": cue["balanced_accuracy"],
        "shortcut_pass": shortcut_pass,
        "summary": str((args.output / "admission_summary.json").relative_to(ROOT)),
    }))


if __name__ == "__main__":
    main()
