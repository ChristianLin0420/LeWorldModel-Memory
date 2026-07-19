#!/usr/bin/env python3
"""OGBench feature-host memory stage.

This is the first full host-stage wrapper for admitted OGBench tasks.  It is
not a native OGBench world-model checkpoint claim.  The frozen host is DINOv2
patch features pooled into a compact host vector; a causal evidence sidecar
writes a residual into that frozen feature interface.  Semantic labels are used
only for the post-training readout.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "ogbench_feature_host_stage_v1"
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


def stable_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dinov2", type=Path, default=DEFAULT_DINOV2)
    parser.add_argument("--torch-home", type=Path, default=DEFAULT_TORCH_HOME)
    parser.add_argument("--prepare-cache", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--train-bases", type=int, default=200)
    parser.add_argument("--validation-bases", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--age", type=int, choices=AGES, default=15)
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--mse-weight", type=float, default=0.05)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in ("output", "dinov2", "torch_home"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    return args


def env_key(env_name: str) -> str:
    return env_name.replace("/", "_")


def cache_path(args: argparse.Namespace) -> Path:
    return args.output / "cache" / env_key(args.env_name) / "features.npz"


def result_dir(args: argparse.Namespace) -> Path:
    return (args.output / env_key(args.env_name)
            / f"age_{int(args.age)}" / f"s{int(args.seed)}")


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
    draw = ImageDraw.Draw(image)
    x, y = [int(v) for v in CARD_POSITIONS[int(label)]]
    draw.rectangle([x - 6, y - 6, x + CARD_SIZE + 6, y + CARD_SIZE + 6],
                   fill=(255, 255, 255), outline=(17, 24, 39), width=2)
    draw.ellipse([x, y, x + CARD_SIZE, y + CARD_SIZE],
                 fill=CARD_COLOR, outline=(17, 24, 39), width=1)
    return np.asarray(image, dtype=np.uint8)


def render_variant(frames: np.ndarray, label: int) -> np.ndarray:
    variant = frames.copy()
    for time in range(1, LAST_CUE_FRAME + 1):
        variant[time] = draw_cue(variant[time], label)
    return variant


def collect_env(args: argparse.Namespace) -> dict[str, np.ndarray]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    import ogbench  # noqa: WPS433

    total = int(args.train_bases + args.validation_bases)
    rng = np.random.default_rng(9109 + int(args.seed))
    env = ogbench.make_env_and_datasets(args.env_name, env_only=True)
    frames = []
    actions = []
    for episode in range(total):
        obs, _ = env.reset(seed=9109 + int(args.seed) + episode)
        del obs
        ep_frames = [resize_frame(env.render())]
        ep_actions = []
        for _ in range(LENGTH - 1):
            action = env.action_space.sample()
            if rng.random() < 0.05:
                action = np.zeros_like(action)
            obs, _, terminated, truncated, _ = env.step(action)
            del obs
            ep_actions.append(np.asarray(action, dtype=np.float32).reshape(-1))
            ep_frames.append(resize_frame(env.render()))
            if terminated or truncated:
                env.reset(seed=9109 + int(args.seed) + episode)
        frames.append(np.stack(ep_frames))
        actions.append(np.stack(ep_actions))
    env.close()
    return {
        "frames": np.stack(frames).astype(np.uint8),
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
    for start in range(0, len(frames), int(batch_size)):
        rows = frames[start:start + int(batch_size)]
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


def split(value: np.ndarray, train_bases: int) -> tuple[np.ndarray, np.ndarray]:
    return value[:train_bases], value[train_bases:]


def prepare_cache(args: argparse.Namespace) -> dict[str, Any]:
    path = cache_path(args)
    if path.is_file() and not args.overwrite_cache:
        return {"status": "exists", "path": str(path.relative_to(ROOT))}
    path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    set_seed(args.seed)
    data = collect_env(args)
    model = load_dinov2(args, device)

    cue_frames = []
    for frames in data["frames"]:
        for label in range(CLASSES):
            cue_frames.append(render_variant(frames, label)[LAST_CUE_FRAME])
    cue = pooled(encode_frames(
        model, np.stack(cue_frames), device, args.feature_batch_size))
    cue = cue.reshape(len(data["frames"]), CLASSES, cue.shape[-1])

    endpoints = {}
    for age in AGES:
        endpoint_frames = np.stack([
            frames[LAST_CUE_FRAME + int(age)] for frames in data["frames"]
        ])
        endpoints[f"endpoint_age{age}"] = pooled(encode_frames(
            model, endpoint_frames, device, args.feature_batch_size))

    train_cue, val_cue = split(cue, args.train_bases)
    payload: dict[str, Any] = {
        "train_cue": train_cue.astype(np.float32),
        "val_cue": val_cue.astype(np.float32),
        "env_name": np.asarray(args.env_name),
        "train_bases": np.asarray(args.train_bases, dtype=np.int64),
        "validation_bases": np.asarray(args.validation_bases, dtype=np.int64),
        "feature_dim": np.asarray(cue.shape[-1], dtype=np.int64),
    }
    for key, value in endpoints.items():
        train_value, val_value = split(value, args.train_bases)
        payload[f"train_{key}"] = train_value.astype(np.float32)
        payload[f"val_{key}"] = val_value.astype(np.float32)
    np.savez_compressed(path, **payload)
    receipt = {
        "schema": "ogbench_feature_host_cache_v1",
        "status": "completed",
        "path": str(path.relative_to(ROOT)),
        "env_name": args.env_name,
        "train_bases": int(args.train_bases),
        "validation_bases": int(args.validation_bases),
        "feature_dim": int(cue.shape[-1]),
        "ages": list(AGES),
        "claim_boundary": "DINO feature-host cache; not a native OGBench world-model checkpoint.",
    }
    (path.parent / "cache_receipt.json").write_text(stable_json(receipt))
    return receipt


class EvidenceWriter(nn.Module):
    def __init__(self, feature_dim: int, dim: int) -> None:
        super().__init__()
        self.cue = nn.Linear(feature_dim, dim)
        self.endpoint = nn.Linear(feature_dim, dim)
        self.trunk = nn.Sequential(
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, 2 * dim),
            nn.SiLU(),
            nn.Linear(2 * dim, dim),
        )
        self.out = nn.Linear(dim, feature_dim)

    def forward(self, cue: torch.Tensor, endpoint: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.cue(cue) + self.endpoint(endpoint)
        memory = self.trunk(hidden)
        residual = self.out(memory)
        return {"host_output": endpoint + residual, "memory_prior": residual}


def expanded(cue: np.ndarray, endpoint: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bases = cue.shape[0]
    labels = np.tile(np.arange(CLASSES, dtype=np.int64), bases)
    cue_flat = cue.reshape(bases * CLASSES, cue.shape[-1])
    endpoint_flat = np.repeat(endpoint, CLASSES, axis=0)
    return cue_flat.astype(np.float32), endpoint_flat.astype(np.float32), labels


def batches(size: int, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    order = rng.permutation(size)
    return [order[start:start + batch_size] for start in range(0, size, batch_size)]


def train_cell(args: argparse.Namespace) -> dict[str, Any]:
    path = cache_path(args)
    if not path.is_file():
        raise FileNotFoundError(f"missing cache: {path}")
    out_dir = result_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    with np.load(path, allow_pickle=False) as data:
        train_cue, train_endpoint, train_y = expanded(
            data["train_cue"], data[f"train_endpoint_age{args.age}"])
        val_cue, val_endpoint, val_y = expanded(
            data["val_cue"], data[f"val_endpoint_age{args.age}"])
    feature_dim = int(train_cue.shape[-1])
    model = EvidenceWriter(feature_dim, args.dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_cue_t = torch.from_numpy(train_cue).to(device)
    train_endpoint_t = torch.from_numpy(train_endpoint).to(device)
    rng = np.random.default_rng(1701 + int(args.seed) + 31 * int(args.age))
    history = []
    model.train()
    for epoch in range(int(args.epochs)):
        losses = []
        for idx in batches(len(train_y), args.batch_size, rng):
            cue_batch = train_cue_t[idx]
            endpoint_batch = train_endpoint_t[idx]
            pred = model(cue_batch, endpoint_batch)["host_output"]
            cosine = 1.0 - F.cosine_similarity(pred, cue_batch, dim=-1).mean()
            mse = F.mse_loss(F.normalize(pred, dim=-1),
                             F.normalize(cue_batch, dim=-1))
            loss = cosine + float(args.mse_weight) * mse
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses))})

    def features(cue_value: np.ndarray, endpoint_value: np.ndarray) -> dict[str, np.ndarray]:
        model.eval()
        outputs, priors = [], []
        with torch.no_grad():
            for start in range(0, len(endpoint_value), args.batch_size):
                stop = start + args.batch_size
                cue_t = torch.from_numpy(cue_value[start:stop]).to(device)
                endpoint_t = torch.from_numpy(endpoint_value[start:stop]).to(device)
                pred = model(cue_t, endpoint_t)
                outputs.append(pred["host_output"].float().cpu().numpy())
                priors.append(pred["memory_prior"].float().cpu().numpy())
        return {
            "host_output": np.concatenate(outputs).astype(np.float32),
            "memory_prior": np.concatenate(priors).astype(np.float32),
        }

    zero_train = np.zeros_like(train_cue)
    zero_val = np.zeros_like(val_cue)
    train_full = features(train_cue, train_endpoint)
    val_full = features(val_cue, val_endpoint)
    val_reset = features(zero_val, val_endpoint)
    train_readout = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0))
    train_readout.fit(train_full["host_output"], train_y)
    prior_readout = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0))
    prior_readout.fit(train_full["memory_prior"], train_y)

    def predict_host(feat: np.ndarray) -> np.ndarray:
        return train_readout.predict(feat).astype(np.int64)

    def predict_prior(feat: np.ndarray) -> np.ndarray:
        return prior_readout.predict(feat).astype(np.int64)

    def metric(name: str, feat: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        pred = predict_host(feat)
        return {
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "confusion_matrix": confusion_matrix(
                y, pred, labels=list(range(CLASSES))).astype(int).tolist(),
            "count": int(len(y)),
            "name": name,
        }

    def prior_metric(name: str, feat: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        pred = predict_prior(feat)
        return {
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "confusion_matrix": confusion_matrix(
                y, pred, labels=list(range(CLASSES))).astype(int).tolist(),
            "count": int(len(y)),
            "name": name,
        }

    no_state = val_endpoint
    result = {
        "schema": "ogbench_feature_host_stage_cell_v1",
        "status": "completed",
        "env_name": args.env_name,
        "age": int(args.age),
        "seed": int(args.seed),
        "feature_dim": feature_dim,
        "dim": int(args.dim),
        "epochs": int(args.epochs),
        "labels_used_for_writer_training": False,
        "labels_used_for_readout": True,
        "claim_boundary": "Frozen DINO feature-host sidecar; not native OGBench world-model planning.",
        "history": history,
        "host_output": {
            "full": metric("full", val_full["host_output"], val_y),
            "reset": metric("reset", val_reset["host_output"], val_y),
            "no_state": metric("no_state", no_state, val_y),
        },
        "memory_prior": {
            "full": prior_metric("full", val_full["memory_prior"], val_y),
            "reset": prior_metric("reset", val_reset["memory_prior"], val_y),
        },
        "gate": {
            "full_minimum": 0.75,
            "control_maximum": 0.30,
        },
    }
    full = result["host_output"]["full"]["balanced_accuracy"]
    reset = result["host_output"]["reset"]["balanced_accuracy"]
    none = result["host_output"]["no_state"]["balanced_accuracy"]
    result["gate"]["pass"] = bool(full >= 0.75 and reset <= 0.30 and none <= 0.30)
    (out_dir / "result.json").write_text(stable_json(result))
    np.savez_compressed(
        out_dir / "features.npz",
        val_y=val_y, full=val_full["host_output"],
        reset=val_reset["host_output"], no_state=no_state,
        prior_full=val_full["memory_prior"],
        prior_reset=val_reset["memory_prior"],
        pred_full=predict_host(val_full["host_output"]),
        pred_reset=predict_host(val_reset["host_output"]),
        pred_no_state=predict_host(no_state),
        pred_prior_full=predict_prior(val_full["memory_prior"]),
        pred_prior_reset=predict_prior(val_reset["memory_prior"]),
    )
    print(json.dumps({
        "env": args.env_name,
        "age": args.age,
        "seed": args.seed,
        "full": full,
        "reset": reset,
        "no_state": none,
        "pass": result["gate"]["pass"],
    }, indent=2), flush=True)
    return result


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output
    rows = []
    for path in sorted(output.glob("*/*/s*/result.json")):
        result = json.loads(path.read_text())
        rows.append(result)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["env_name"], int(row["age"])), []).append(row)
    summary_rows = []
    for (name, age), values in sorted(grouped.items()):
        full = [v["host_output"]["full"]["balanced_accuracy"] for v in values]
        reset = [v["host_output"]["reset"]["balanced_accuracy"] for v in values]
        none = [v["host_output"]["no_state"]["balanced_accuracy"] for v in values]
        prior = [v["memory_prior"]["full"]["balanced_accuracy"] for v in values]
        summary_rows.append({
            "env_name": name,
            "age": age,
            "seeds": [int(v["seed"]) for v in values],
            "host_full_mean": float(np.mean(full)),
            "host_reset_mean": float(np.mean(reset)),
            "host_no_state_mean": float(np.mean(none)),
            "prior_full_mean": float(np.mean(prior)),
            "pass_count": int(sum(bool(v["gate"]["pass"]) for v in values)),
            "seed_count": int(len(values)),
            "all_pass": bool(all(bool(v["gate"]["pass"]) for v in values)),
        })
    summary = {
        "schema": "ogbench_feature_host_stage_summary_v1",
        "status": "completed" if rows else "empty",
        "claim_boundary": "Frozen DINO feature-host sidecar; not native OGBench world-model planning.",
        "rows": summary_rows,
        "cell_count": int(len(rows)),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(stable_json(summary))
    print(stable_json(summary), flush=True)
    return summary


def main() -> None:
    args = resolve_args(parse_args())
    args.output.mkdir(parents=True, exist_ok=True)
    if args.prepare_cache:
        print(stable_json(prepare_cache(args)), flush=True)
        return
    if args.aggregate:
        aggregate(args)
        return
    train_cell(args)


if __name__ == "__main__":
    main()
