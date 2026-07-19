#!/usr/bin/env python3
"""Stage-G admission deck for official DINO-WM Wall.

This is deliberately narrower than the PointMaze formal runner.  It prepares
the Wall expansion only up to the fail-closed admission gates:

* transient cue is visible and decodable at cue time;
* the same label is not decodable from post-cue endpoint visual features;
* actions/proprio do not carry label shortcuts; and
* generated counterfactual cue variants only modify the declared cue window.

Carrier training should only be launched after this script reports admitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/dinowm_wall_audit_v1/stage_g_admission"
DEFAULT_DATA = ROOT / "outputs/dinowm_wall_audit_v1/data/wall_single"
DEFAULT_CHECKPOINT = ROOT / "outputs/dinowm_wall_audit_v1/checkpoint/model_latest.pth"
DEFAULT_HYDRA = ROOT / "outputs/dinowm_wall_audit_v1/checkpoint/hydra.yaml"
DEFAULT_VENDOR = ROOT / "outputs/dinowm_native_pusht_audit_v1/vendor/dino_wm"
DEFAULT_DINOV2 = ROOT / "outputs/dinowm_native_pusht_audit_v1/vendor/dinov2"
DEFAULT_TORCH_HOME = ROOT / "outputs/dinowm_native_pusht_audit_v1/torch_home"

AGES = (4, 8, 15)
CLASSES = 4
LENGTH = 20
LAST_CUE_FRAME = 3
CARD_POSITIONS = np.asarray([
    [16, 16],
    [176, 16],
    [16, 176],
    [176, 176],
], dtype=np.int64)
CARD_COLOR = (230, 57, 70)
CARD_SIZE = 28


def stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, indent=2) + "\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--hydra", type=Path, default=DEFAULT_HYDRA)
    parser.add_argument("--vendor", type=Path, default=DEFAULT_VENDOR)
    parser.add_argument("--dinov2", type=Path, default=DEFAULT_DINOV2)
    parser.add_argument("--torch-home", type=Path, default=DEFAULT_TORCH_HOME)
    parser.add_argument("--train-bases", type=int, default=300)
    parser.add_argument("--validation-bases", type=int, default=120)
    parser.add_argument("--seed", type=int, default=871003)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_wall_arrays(data_root: Path) -> dict[str, torch.Tensor]:
    required = ["states.pth", "actions.pth", "door_locations.pth",
                "wall_locations.pth"]
    for name in required:
        if not (data_root / name).is_file():
            raise FileNotFoundError(data_root / name)
    states = torch.load(data_root / "states.pth", map_location="cpu",
                        weights_only=False).float()
    actions = torch.load(data_root / "actions.pth", map_location="cpu",
                         weights_only=False).float()
    door = torch.load(data_root / "door_locations.pth", map_location="cpu",
                      weights_only=False).float()
    wall = torch.load(data_root / "wall_locations.pth", map_location="cpu",
                      weights_only=False).float()
    if states.shape[:2] != actions.shape[:2] or states.shape[-1] != 2 \
            or actions.shape[-1] != 2:
        raise RuntimeError("unexpected Wall state/action shape")
    return {"states": states, "actions": actions, "door": door, "wall": wall}


def select_bases(total: int, train_count: int, validation_count: int,
                 seed: int) -> dict[str, np.ndarray]:
    if train_count + validation_count > total:
        raise ValueError("requested more Wall episodes than available")
    rng = np.random.default_rng(seed)
    order = rng.permutation(total)
    return {
        "train": np.sort(order[:train_count]).astype(np.int64),
        "validation": np.sort(
            order[train_count:train_count + validation_count]).astype(np.int64),
    }


def to_uint8_hwc(tensor: torch.Tensor) -> np.ndarray:
    value = tensor.detach().cpu().numpy()
    if value.shape[0] != 3:
        raise RuntimeError("Wall frame must be CHW")
    value = np.clip(value, 0, 255).astype(np.uint8)
    return np.transpose(value, (1, 2, 0)).copy()


def draw_cue(frame: np.ndarray, label: int) -> np.ndarray:
    image = Image.fromarray(frame.copy())
    draw = ImageDraw.Draw(image)
    x, y = [int(v) for v in CARD_POSITIONS[int(label)]]
    draw.rectangle(
        [x - 6, y - 6, x + CARD_SIZE + 6, y + CARD_SIZE + 6],
        fill=(255, 255, 255), outline=(17, 24, 39), width=2)
    draw.ellipse(
        [x, y, x + CARD_SIZE, y + CARD_SIZE],
        fill=CARD_COLOR, outline=(17, 24, 39), width=1)
    return np.asarray(image, dtype=np.uint8)


def episode_frames(data_root: Path, episode: int) -> np.ndarray:
    path = data_root / "obses" / f"episode_{int(episode):03d}.pth"
    frames = torch.load(path, map_location="cpu", weights_only=False).float()
    if tuple(frames.shape[:2]) != (51, 3):
        raise RuntimeError(f"unexpected Wall observation shape: {frames.shape}")
    return np.stack([to_uint8_hwc(frames[t]) for t in range(LENGTH)])


def render_variant(base_frames: np.ndarray, label: int) -> np.ndarray:
    variant = base_frames.copy()
    for t in range(1, LAST_CUE_FRAME + 1):
        variant[t] = draw_cue(variant[t], label)
    return variant


def action_blocks(actions: torch.Tensor, indices: np.ndarray) -> np.ndarray:
    values = actions[indices].numpy().astype(np.float32)
    mean = actions.mean(dim=(0, 1)).numpy().astype(np.float32)
    std = np.maximum(actions.std(dim=(0, 1)).numpy().astype(np.float32),
                     1e-6)
    norm = (values - mean) / std
    blocks = []
    for t in range(LENGTH - 1):
        block = norm[:, t:t + 5].reshape(len(indices), -1)
        blocks.append(block)
    return np.stack(blocks, axis=1).astype(np.float32)


def normalized_proprio(states: torch.Tensor, indices: np.ndarray) -> np.ndarray:
    values = states[indices, :LENGTH].numpy().astype(np.float32)
    mean = states.mean(dim=(0, 1)).numpy().astype(np.float32)
    std = np.maximum(states.std(dim=(0, 1)).numpy().astype(np.float32), 1e-6)
    return ((values - mean) / std).astype(np.float32)


def load_dinov2(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    os.environ["TORCH_HOME"] = str(args.torch_home.resolve())
    sys.path.insert(0, str(args.dinov2.resolve()))
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
        patches = model.forward_features(tensor)["x_norm_patchtokens"]
        outputs.append(patches.float().cpu().numpy())
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


def labels_for(base_count: int) -> np.ndarray:
    return np.tile(np.arange(CLASSES, dtype=np.int64), base_count)


def expand_base_feature(feature: np.ndarray) -> np.ndarray:
    return np.repeat(feature, CLASSES, axis=0)


def fit_predict(train_x: np.ndarray, train_y: np.ndarray,
                val_x: np.ndarray) -> np.ndarray:
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=4000,
                           random_state=0))
    clf.fit(train_x, train_y)
    return clf.predict(val_x).astype(np.int64)


def classify_record(pred: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "confusion_matrix": confusion_matrix(
            truth, pred, labels=list(range(CLASSES))).astype(int).tolist(),
        "count": int(len(truth)),
    }


def build_split(args: argparse.Namespace, model: torch.nn.Module,
                device: torch.device, indices: np.ndarray,
                arrays: dict[str, torch.Tensor],
                split: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    base_frames = [episode_frames(args.data, int(ep)) for ep in indices]
    cue_frames = []
    endpoint_frames = {age: [] for age in AGES}
    outside = pairwise = post_cue = 0
    min_changed = None
    for frames in base_frames:
        variants = [render_variant(frames, label) for label in range(CLASSES)]
        for label, variant in enumerate(variants):
            cue_frames.append(variant[LAST_CUE_FRAME])
            changed = int(np.count_nonzero(
                variant[1:LAST_CUE_FRAME + 1]
                != frames[1:LAST_CUE_FRAME + 1]))
            min_changed = changed if min_changed is None else min(min_changed, changed)
            post_cue += int(np.count_nonzero(variant[LAST_CUE_FRAME + 1:] !=
                                             frames[LAST_CUE_FRAME + 1:]))
            # The declared cue mask is the four possible card rectangles.
            mask = np.zeros(frames.shape[1:3], dtype=bool)
            for x, y in CARD_POSITIONS:
                mask[max(0, y - 6):min(224, y + CARD_SIZE + 7),
                     max(0, x - 6):min(224, x + CARD_SIZE + 7)] = True
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
            endpoint = LAST_CUE_FRAME + age
            endpoint_frames[age].append(frames[endpoint])
    cue = pooled(encode_frames(model, np.stack(cue_frames), device,
                               args.batch_size))
    endpoint = {
        age: pooled(encode_frames(model, np.stack(frames), device,
                                  args.batch_size))
        for age, frames in endpoint_frames.items()
    }
    actions = action_blocks(arrays["actions"], indices)
    proprio = normalized_proprio(arrays["states"], indices)
    records = {
        "cue": cue,
        "endpoint": endpoint,
        "actions": actions,
        "proprio": proprio,
    }
    audit = {
        "split": split,
        "base_count": int(len(indices)),
        "expanded_count": int(len(indices) * CLASSES),
        "outside_declared_mask_changed_pixels": int(outside),
        "pairwise_outside_mask_changed_pixels": int(pairwise),
        "post_cue_differing_pixels": int(post_cue),
        "minimum_changed_cue_pixels_any_label": int(min_changed or 0),
    }
    return records, audit


def main() -> None:
    args = parse_args()
    for name in ("output", "data", "checkpoint", "hydra", "vendor",
                 "dinov2", "torch_home"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    arrays = load_wall_arrays(args.data)
    selections = select_bases(
        int(arrays["states"].shape[0]), args.train_bases,
        args.validation_bases, args.seed)
    model = load_dinov2(args, device)
    train, train_audit = build_split(
        args, model, device, selections["train"], arrays, "train")
    validation, val_audit = build_split(
        args, model, device, selections["validation"], arrays, "validation")

    y_train = labels_for(args.train_bases)
    y_val = labels_for(args.validation_bases)
    cue_pred = fit_predict(train["cue"], y_train, validation["cue"])
    cue_record = classify_record(cue_pred, y_val)
    shortcuts = {}
    for age in AGES:
        endpoint_pred = fit_predict(
            expand_base_feature(train["endpoint"][age]), y_train,
            expand_base_feature(validation["endpoint"][age]))
        endpoint = LAST_CUE_FRAME + age
        action_pred = fit_predict(
            expand_base_feature(train["actions"][:, endpoint - 1]),
            y_train,
            expand_base_feature(validation["actions"][:, endpoint - 1]))
        proprio_pred = fit_predict(
            expand_base_feature(train["proprio"][:, endpoint - 1]),
            y_train,
            expand_base_feature(validation["proprio"][:, endpoint - 1]))
        shortcuts[str(age)] = {
            "no_cue_visual_endpoint": classify_record(endpoint_pred, y_val),
            "action_only": classify_record(action_pred, y_val),
            "proprio_only": classify_record(proprio_pred, y_val),
        }

    shortcut_max = 1.0 / CLASSES + 0.05
    cue_pass = cue_record["balanced_accuracy"] >= 0.75
    shortcut_pass = all(
        value["balanced_accuracy"] <= shortcut_max
        for record in shortcuts.values()
        for value in record.values())
    counterfactual_pass = all(
        item["outside_declared_mask_changed_pixels"] == 0
        and item["pairwise_outside_mask_changed_pixels"] == 0
        and item["post_cue_differing_pixels"] == 0
        for item in (train_audit, val_audit))
    admitted = bool(cue_pass and shortcut_pass and counterfactual_pass)

    sample = np.concatenate([
        render_variant(episode_frames(args.data, int(selections["validation"][0])),
                       label)[LAST_CUE_FRAME]
        for label in range(CLASSES)
    ], axis=1)
    Image.fromarray(sample).save(args.output / "wall_cue_variants.png")
    np.savez_compressed(
        args.output / "selection.npz",
        train=selections["train"], validation=selections["validation"])
    summary = {
        "schema": "dinowm_wall_stage_g_admission_v1",
        "status": "admitted" if admitted else "stopped_fail_closed",
        "admitted": admitted,
        "task": {
            "environment": "DINO-WM Wall",
            "classes": CLASSES,
            "cue_frames": [1, 2, 3],
            "evidence_ages": list(AGES),
            "endpoint_frames": [LAST_CUE_FRAME + age for age in AGES],
            "predictor_context": "one previous sampled frame",
        },
        "assets": {
            "checkpoint": {
                "path": str(args.checkpoint.relative_to(ROOT)),
                "sha256": sha256_file(args.checkpoint),
                "size": args.checkpoint.stat().st_size,
            },
            "hydra": {
                "path": str(args.hydra.relative_to(ROOT)),
                "sha256": sha256_file(args.hydra),
                "size": args.hydra.stat().st_size,
            },
            "dataset": {
                "path": str(args.data.relative_to(ROOT)),
                "episodes": int(arrays["states"].shape[0]),
                "frames_per_episode": int(arrays["states"].shape[1] + 1),
            },
        },
        "selection": {
            "seed": int(args.seed),
            "train_bases": int(args.train_bases),
            "validation_bases": int(args.validation_bases),
            "expanded_train": int(args.train_bases * CLASSES),
            "expanded_validation": int(args.validation_bases * CLASSES),
        },
        "cue_encoding": {
            **cue_record,
            "minimum": 0.75,
            "pass": bool(cue_pass),
        },
        "shortcuts": shortcuts,
        "shortcut_threshold": shortcut_max,
        "shortcut_pass": bool(shortcut_pass),
        "counterfactual_audit": {
            "train": train_audit,
            "validation": val_audit,
            "pass": bool(counterfactual_pass),
        },
        "artifacts": {
            "selection": "selection.npz",
            "cue_variants": "wall_cue_variants.png",
        },
    }
    (args.output / "admission_summary.json").write_text(stable_json(summary))
    print(stable_json({
        "status": summary["status"],
        "admitted": admitted,
        "cue_bacc": cue_record["balanced_accuracy"],
        "shortcut_pass": shortcut_pass,
        "summary": str((args.output / "admission_summary.json").relative_to(ROOT)),
    }))


if __name__ == "__main__":
    main()
