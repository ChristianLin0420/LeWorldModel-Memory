#!/usr/bin/env python3
"""Autonomous Masked-Evidence JEPA Memory stage on OGBench renders.

This runner is intentionally scoped as a first method-stage implementation.
It removes the manually supplied cue-feature sidecar used by the feature-host
capacity experiment.  The model sees a causal rendered history with transient
cue cards and must predict a stop-gradient latent target built from masked
past evidence.  Cue labels are used only for post-hoc readout/evaluation.

Claim boundary:
  * self-supervised latent target prediction, not RGB decoding;
  * compact slot memory, not KV-cache or recurrent GRU/LSTM state;
  * OGBench render-stage method probe, not native LeWM/DINO-WM planning.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import subprocess
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.linear_model import RidgeClassifier  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, confusion_matrix  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "masked_evidence_jepa_ogbench_v1"


def _env_int_tuple(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = os.environ.get(name)
    if not raw:
        return default
    return tuple(int(v) for v in raw.replace(",", " ").split())


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else int(default)


# Horizon constants are overridable via environment variables so the age-scaling
# study (reviewer concern 8) can re-render longer episodes and evaluate delays
# far beyond 15 without touching call sites:
#   PAPER_C_AGES="4 8 16 32 64 128"  PAPER_C_LENGTH=140
AGES = _env_int_tuple("PAPER_C_AGES", (4, 8, 15))
CLASSES = 4
LENGTH = _env_int("PAPER_C_LENGTH", 22)
LAST_CUE_FRAME = _env_int("PAPER_C_LAST_CUE_FRAME", 3)
MAX_ENDPOINT = LAST_CUE_FRAME + max(AGES)
MAX_CONTEXT = MAX_ENDPOINT + 1
PALETTE = np.asarray(
    [[230, 57, 70], [46, 204, 113], [52, 152, 219], [245, 203, 92]],
    dtype=np.uint8,
)


def stable_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def env_key(env_name: str) -> str:
    return env_name.replace("/", "_")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--env-name", default="pointmaze-large-navigate-v0")
    parser.add_argument("--prepare-cache", action="store_true")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--episodes", type=int, default=384)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--age", type=int, choices=AGES, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--dim", type=int, default=160)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument(
        "--chunk",
        type=int,
        default=0,
        help="Streaming chunk size K (bounded eviction). 0 = one-shot full-prefix attention.",
    )
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--temperature", type=float, default=0.08)
    parser.add_argument("--cos-weight", type=float, default=0.35)
    parser.add_argument("--std-weight", type=float, default=0.05)
    parser.add_argument("--temporal-drop", type=float, default=0.12)
    parser.add_argument("--patch-drop", type=float, default=0.20)
    parser.add_argument(
        "--cue-mode",
        default="color",
        choices=CUE_MODES,
        help="Cue family. 'color' is the original; others decorrelate colour from the label.",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def cache_path(args: argparse.Namespace) -> Path:
    return args.output / "cache" / env_key(args.env_name) / "render_cache.npz"


def result_dir(args: argparse.Namespace) -> Path:
    return args.output / env_key(args.env_name) / f"age_{args.age}" / f"s{args.seed}"


def cue_layout(img_size: int) -> tuple[np.ndarray, int]:
    card = max(8, int(round(img_size * 0.17)))
    margin = max(4, int(round(img_size * 0.08)))
    positions = np.asarray(
        [
            [margin, margin],
            [img_size - margin - card, margin],
            [margin, img_size - margin - card],
            [img_size - margin - card, img_size - margin - card],
        ],
        dtype=np.int64,
    )
    return positions, card


def resize_frame(frame: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(
        Image.fromarray(frame.astype(np.uint8)).resize(
            (int(size), int(size)), Image.Resampling.BILINEAR
        ),
        dtype=np.uint8,
    )


def draw_cue(frame: np.ndarray, label: int, position: int) -> np.ndarray:
    image = Image.fromarray(frame.copy())
    draw = ImageDraw.Draw(image)
    positions, card = cue_layout(frame.shape[0])
    x, y = [int(v) for v in positions[int(position)]]
    color = tuple(int(v) for v in PALETTE[int(label)])
    pad = max(2, card // 6)
    draw.rectangle(
        [x - pad, y - pad, x + card + pad, y + card + pad],
        fill=(255, 255, 255),
        outline=(17, 24, 39),
        width=max(1, card // 9),
    )
    draw.ellipse(
        [x, y, x + card, y + card],
        fill=color,
        outline=(17, 24, 39),
        width=max(1, card // 12),
    )
    return np.asarray(image, dtype=np.uint8)


def inject_cue_sequence(frames: np.ndarray, label: int, position: int) -> np.ndarray:
    out = frames.copy()
    for time in range(1, LAST_CUE_FRAME + 1):
        out[time] = draw_cue(out[time], int(label), int(position))
    return out


# --------------------------------------------------------------------------- #
# Shortcut-breaking cue families (reviewer concern 4).
# The original ``color`` cue makes label recoverable from pooled color alone.
# These variants decorrelate colour from the label so the memory must retain
# shape identity, spatial relation, or count rather than a single hue.
# --------------------------------------------------------------------------- #
CUE_MODES = ("color", "shape_random_color", "relational", "palette_holdout", "two_cue_overwrite")
# A large held-out palette so appearance is not a 4-way colour code.
_RANDOM_PALETTE = np.asarray(
    [
        [230, 57, 70], [46, 204, 113], [52, 152, 219], [245, 203, 92],
        [155, 89, 182], [26, 188, 156], [231, 126, 34], [149, 165, 166],
        [211, 84, 0], [22, 160, 133], [192, 57, 43], [41, 128, 185],
    ],
    dtype=np.uint8,
)


def _draw_shape(draw, shape: int, box, color) -> None:
    x0, y0, x1, y1 = box
    outline = (17, 24, 39)
    w = max(1, (x1 - x0) // 10)
    if shape == 0:  # circle
        draw.ellipse([x0, y0, x1, y1], fill=color, outline=outline, width=w)
    elif shape == 1:  # triangle
        draw.polygon([(x0 + (x1 - x0) // 2, y0), (x0, y1), (x1, y1)], fill=color, outline=outline)
    elif shape == 2:  # square
        draw.rectangle([x0, y0, x1, y1], fill=color, outline=outline, width=w)
    else:  # diamond
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        draw.polygon([(cx, y0), (x1, cy), (cx, y1), (x0, cy)], fill=color, outline=outline)


def draw_cue_shape(frame: np.ndarray, shape: int, position: int, color) -> np.ndarray:
    """Draw a shape (label = shape identity) with an arbitrary colour."""
    image = Image.fromarray(frame.copy())
    draw = ImageDraw.Draw(image)
    positions, card = cue_layout(frame.shape[0])
    x, y = [int(v) for v in positions[int(position)]]
    pad = max(2, card // 6)
    draw.rectangle(
        [x - pad, y - pad, x + card + pad, y + card + pad],
        fill=(255, 255, 255),
        outline=(17, 24, 39),
        width=max(1, card // 9),
    )
    _draw_shape(draw, int(shape), (x, y, x + card, y + card), tuple(int(v) for v in color))
    return np.asarray(image, dtype=np.uint8)


def _episode_rng(episode: int, salt: int) -> np.random.Generator:
    return np.random.default_rng(9_100_003 + int(episode) * 131 + int(salt))


def inject_cue_sequence_mode(
    frames: np.ndarray, label: int, position: int, *, mode: str, episode: int
) -> np.ndarray:
    """Cue injection dispatcher. ``color`` reproduces the original overlay."""
    if mode == "color":
        return inject_cue_sequence(frames, label, position)

    out = frames.copy()
    rng = _episode_rng(episode, 7)
    if mode == "shape_random_color":
        # Label = shape identity; colour is random and label-independent.
        color = _RANDOM_PALETTE[int(rng.integers(0, len(_RANDOM_PALETTE)))]
        for time in range(1, LAST_CUE_FRAME + 1):
            out[time] = draw_cue_shape(out[time], int(label), int(position), color)
        return out
    if mode == "palette_holdout":
        # Colour drawn from a large palette, decorrelated from the 4-way label,
        # while shape encodes the label.
        color = _RANDOM_PALETTE[int(rng.integers(4, len(_RANDOM_PALETTE)))]
        for time in range(1, LAST_CUE_FRAME + 1):
            out[time] = draw_cue_shape(out[time], int(label), int(position), color)
        return out
    if mode == "relational":
        # Two shapes; label in {0,1,2,3} encodes the relation (which of two
        # slots holds the marker, x2 orderings). Colour is random.
        left_first = int(label) % 2
        pair = sorted([int(position), int((position + 1 + int(label) // 2) % 4)])
        marker = pair[0] if left_first else pair[1]
        other = pair[1] if left_first else pair[0]
        c1 = _RANDOM_PALETTE[int(rng.integers(0, len(_RANDOM_PALETTE)))]
        c2 = _RANDOM_PALETTE[int(rng.integers(0, len(_RANDOM_PALETTE)))]
        for time in range(1, LAST_CUE_FRAME + 1):
            f = draw_cue_shape(out[time], 0, marker, c1)
            f = draw_cue_shape(f, 2, other, c2)
            out[time] = f
        return out
    if mode == "two_cue_overwrite":
        # An early distractor cue is later overwritten; the LATER cue is the
        # answer, so a model that keeps only the first cue fails.
        distractor = int(rng.integers(0, CLASSES))
        for time in range(1, LAST_CUE_FRAME + 1):
            out[time] = draw_cue(out[time], distractor, int(position))
        for time in range(LAST_CUE_FRAME + 1, min(2 * LAST_CUE_FRAME + 1, frames.shape[0])):
            color = _RANDOM_PALETTE[int(rng.integers(0, len(_RANDOM_PALETTE)))]
            out[time] = draw_cue_shape(out[time], int(label), int(position), color)
        return out
    raise ValueError(f"unknown cue mode {mode!r}; choices={CUE_MODES}")


def crop_fixed(frame: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    crop = np.zeros((size, size, 3), dtype=np.uint8)
    h, w = frame.shape[:2]
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(w, int(x) + size), min(h, int(y) + size)
    if x1 <= x0 or y1 <= y0:
        return crop
    crop[: y1 - y0, : x1 - x0] = frame[y0:y1, x0:x1]
    return crop


def evidence_panel(frame: np.ndarray) -> np.ndarray:
    positions, card = cue_layout(frame.shape[0])
    crop_size = int(card + max(4, card // 2))
    pad = (crop_size - card) // 2
    crops = [
        crop_fixed(frame, int(x) - pad, int(y) - pad, crop_size)
        for x, y in positions
    ]
    top = np.concatenate(crops[:2], axis=1)
    bottom = np.concatenate(crops[2:], axis=1)
    return np.concatenate([top, bottom], axis=0).astype(np.uint8)


def prepare_cache(args: argparse.Namespace) -> dict[str, Any]:
    path = cache_path(args)
    if path.is_file() and not args.overwrite_cache:
        return {"status": "exists", "path": str(path.relative_to(ROOT))}
    path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MUJOCO_GL", "egl")
    import ogbench  # noqa: WPS433

    rng = np.random.default_rng(31_337 + int(args.seed))
    env = ogbench.make_env_and_datasets(args.env_name, env_only=True)
    frames, actions = [], []
    try:
        env.action_space.seed(71_001 + int(args.seed))
    except Exception:
        pass
    for episode in range(int(args.episodes)):
        obs, _ = env.reset(seed=71_001 + int(args.seed) + episode)
        del obs
        episode_frames = [resize_frame(env.render(), args.img_size)]
        episode_actions = []
        for _ in range(LENGTH - 1):
            action = env.action_space.sample()
            if rng.random() < 0.05:
                action = np.zeros_like(action)
            obs, _, terminated, truncated, _ = env.step(action)
            del obs
            episode_actions.append(np.asarray(action, dtype=np.float32).reshape(-1))
            episode_frames.append(resize_frame(env.render(), args.img_size))
            if terminated or truncated:
                env.reset(seed=81_001 + int(args.seed) + episode)
        frames.append(np.stack(episode_frames).astype(np.uint8))
        actions.append(np.stack(episode_actions).astype(np.float32))
    env.close()

    labels = np.tile(np.arange(CLASSES, dtype=np.int64), int(np.ceil(args.episodes / CLASSES)))[: args.episodes]
    positions = np.tile(np.arange(CLASSES, dtype=np.int64), int(np.ceil(args.episodes / CLASSES)))[: args.episodes]
    rng.shuffle(labels)
    rng.shuffle(positions)
    np.savez_compressed(
        path,
        frames=np.stack(frames),
        actions=np.stack(actions),
        cue_labels=labels.astype(np.int64),
        cue_positions=positions.astype(np.int64),
        env_name=np.asarray(args.env_name),
        img_size=np.asarray(args.img_size, dtype=np.int64),
        length=np.asarray(LENGTH, dtype=np.int64),
        seed=np.asarray(args.seed, dtype=np.int64),
    )
    receipt = {
        "schema": "masked_evidence_jepa_ogbench_cache_v1",
        "status": "completed",
        "path": str(path.relative_to(ROOT)),
        "env_name": args.env_name,
        "episodes": int(args.episodes),
        "img_size": int(args.img_size),
        "cue_labels_used_for_training_loss": False,
        "claim_boundary": "render-cache for autonomous masked-evidence JEPA; not a native world-model checkpoint.",
    }
    (path.parent / "cache_receipt.json").write_text(stable_json(receipt))
    return receipt


class MaskedEvidenceDataset(Dataset):
    def __init__(
        self,
        archive: Path,
        *,
        age: int,
        split: str,
        seed: int,
        validation_fraction: float,
        variant: str = "full",
        augment: bool = False,
        temporal_drop: float = 0.0,
        patch_drop: float = 0.0,
    ) -> None:
        with np.load(archive, allow_pickle=False) as data:
            self.frames = data["frames"]
            self.actions = data["actions"]
            self.labels = data["cue_labels"]
            self.positions = data["cue_positions"]
        self.age = int(age)
        self.endpoint = LAST_CUE_FRAME + self.age
        self.variant = str(variant)
        self.augment = bool(augment)
        self.temporal_drop = float(temporal_drop)
        self.patch_drop = float(patch_drop)
        rng = np.random.default_rng(97_101 + int(seed))
        order = rng.permutation(len(self.frames))
        val_count = max(CLASSES, int(round(len(order) * float(validation_fraction))))
        self.indices = order[:-val_count] if split == "train" else order[-val_count:]
        if self.endpoint >= self.frames.shape[1]:
            raise ValueError(f"age {age} exceeds cached sequence length")

    def __len__(self) -> int:
        return int(len(self.indices))

    def _maybe_mask_patch(self, frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if not self.augment or rng.random() >= self.patch_drop:
            return frame
        out = frame.copy()
        size = max(6, frame.shape[0] // 5)
        x = int(rng.integers(0, max(1, frame.shape[1] - size)))
        y = int(rng.integers(0, max(1, frame.shape[0] - size)))
        out[y:y + size, x:x + size] = np.asarray([18, 18, 18], dtype=np.uint8)
        return out

    def _valid_times(self, rng: np.random.Generator) -> list[int]:
        if self.variant == "no_state":
            return list(range(max(0, self.endpoint - 3), self.endpoint + 1))
        times = list(range(0, self.endpoint + 1))
        if self.augment and self.temporal_drop > 0:
            kept = []
            for time in times:
                if time == self.endpoint or rng.random() > self.temporal_drop:
                    kept.append(time)
            if not any(1 <= t <= LAST_CUE_FRAME for t in kept):
                kept.append(int(rng.integers(1, LAST_CUE_FRAME + 1)))
            times = sorted(set(kept))
        return times

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        episode = int(self.indices[item])
        label = int(self.labels[episode])
        position = int(self.positions[episode])
        rng = np.random.default_rng(10_000_019 + episode + 101 * self.age)
        clean = self.frames[episode].copy()
        full = inject_cue_sequence(clean, label, position)
        source = clean if self.variant in {"reset", "no_state"} else full
        times = self._valid_times(rng)
        frame_tokens = np.zeros(
            (MAX_CONTEXT, clean.shape[-3], clean.shape[-2], clean.shape[-1]),
            dtype=np.uint8,
        )
        action_tokens = np.zeros((MAX_CONTEXT, self.actions.shape[-1]), dtype=np.float32)
        time_tokens = np.zeros((MAX_CONTEXT, 1), dtype=np.float32)
        valid = np.zeros((MAX_CONTEXT,), dtype=np.float32)
        for slot, time in enumerate(times[:MAX_CONTEXT]):
            frame_tokens[slot] = self._maybe_mask_patch(source[time], rng)
            if time > 0:
                action_tokens[slot] = self.actions[episode, time - 1]
            time_tokens[slot, 0] = float(time) / float(LENGTH - 1)
            valid[slot] = 1.0
        target_frame = full[LAST_CUE_FRAME]
        panel = evidence_panel(target_frame)
        return {
            "frames": torch.from_numpy(frame_tokens.astype(np.float32) / 255.0).permute(0, 3, 1, 2),
            "actions": torch.from_numpy(action_tokens),
            "times": torch.from_numpy(time_tokens),
            "valid": torch.from_numpy(valid),
            "target_panel": torch.from_numpy(panel.astype(np.float32) / 255.0).permute(2, 0, 1),
            "label": torch.tensor(label, dtype=torch.long),
        }


class FrameEncoder(nn.Module):
    def __init__(self, dim: int, img_size: int) -> None:
        super().__init__()
        final = int(img_size // 8)
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(96 * final * final, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FixedPanelTargetEncoder(nn.Module):
    def __init__(self, dim: int, grid: int = 8) -> None:
        super().__init__()
        raw_dim = 7 * grid * grid
        self.grid = int(grid)
        projection = torch.randn(raw_dim, dim) / float(raw_dim) ** 0.5
        self.register_buffer("projection", projection)

    def forward(self, panel: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(panel, (self.grid, self.grid))
        centered = pooled - pooled.mean(dim=(2, 3), keepdim=True)
        saturation = pooled.max(dim=1, keepdim=True).values - pooled.min(
            dim=1, keepdim=True
        ).values
        raw = torch.cat([pooled - 0.5, centered, saturation], dim=1).flatten(1)
        return F.normalize(raw @ self.projection, dim=-1)


class SlotMemory(nn.Module):
    def __init__(self, dim: int, slots: int, heads: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(slots, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.SiLU(), nn.Linear(4 * dim, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self, tokens: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.query.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        slots, weights = self.attn(
            query,
            tokens,
            tokens,
            key_padding_mask=(valid < 0.5),
            need_weights=True,
            average_attn_weights=False,
        )
        slots = self.norm1(slots + query)
        slots = self.norm2(slots + self.ff(slots))
        return slots, weights.mean(dim=1)


class StreamingSlotMemory(nn.Module):
    """Bounded streaming slot writer: ``M_t = f(M_{t-1}, Z_{t-K+1:t})``.

    The writer maintains a fixed ``S x D`` slot state and ingests the token
    stream in consecutive chunks of at most ``chunk`` (=K) tokens. Each update
    lets the slot queries attend over ``[previous slots ; new chunk tokens]``
    only; the chunk tokens are then discarded. Peak memory and per-step compute
    are therefore constant in sequence length. With ``chunk <= 0`` (or
    ``chunk >= steps``) it reduces exactly to one-shot :class:`SlotMemory` over
    the full prefix, which we keep as an ablation.
    """

    def __init__(self, dim: int, slots: int, heads: int, chunk: int = 4) -> None:
        super().__init__()
        self.slots = int(slots)
        self.chunk = int(chunk)
        self.query = nn.Parameter(torch.randn(slots, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.SiLU(), nn.Linear(4 * dim, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def _update(
        self, state: torch.Tensor, tokens: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        kv = torch.cat([state, tokens], dim=1)
        slot_valid = torch.ones(
            state.shape[0], state.shape[1], device=state.device, dtype=valid.dtype
        )
        kv_valid = torch.cat([slot_valid, valid], dim=1)
        attended, _ = self.attn(
            state, kv, kv, key_padding_mask=(kv_valid < 0.5), need_weights=False
        )
        updated = self.norm1(attended + state)
        updated = self.norm2(updated + self.ff(updated))
        return updated

    def forward(
        self, tokens: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, steps = tokens.shape[0], tokens.shape[1]
        state = self.query.unsqueeze(0).expand(bsz, -1, -1).contiguous()
        chunk = self.chunk if self.chunk > 0 else steps
        for start in range(0, steps, chunk):
            end = min(steps, start + chunk)
            chunk_valid = valid[:, start:end]
            if float(chunk_valid.sum()) == 0.0:
                continue
            state = self._update(state, tokens[:, start:end], chunk_valid)
        return state, None


class MaskedEvidenceJEPA(nn.Module):
    def __init__(
        self,
        *,
        img_size: int,
        action_dim: int,
        dim: int,
        slots: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.frame = FrameEncoder(dim, img_size)
        self.target = FixedPanelTargetEncoder(dim)
        self.action = nn.Sequential(
            nn.Linear(action_dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.time = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.memory = SlotMemory(dim, slots, heads)
        self.pred = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 2 * dim),
            nn.SiLU(),
            nn.Linear(2 * dim, dim),
        )

    def encode_context(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        frames = batch["frames"]
        bsz, steps = frames.shape[:2]
        flat = frames.reshape(bsz * steps, *frames.shape[2:])
        tokens = self.frame(flat).reshape(bsz, steps, -1)
        tokens = tokens + self.action(batch["actions"]) + self.time(batch["times"])
        slots, _ = self.memory(tokens, batch["valid"])
        return F.normalize(slots.mean(dim=1), dim=-1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        memory = self.encode_context(batch)
        pred = F.normalize(self.pred(memory), dim=-1)
        with torch.no_grad():
            target = self.target(batch["target_panel"])
        return {"memory": memory, "pred": pred, "target": target}


def info_nce(pred: torch.Tensor, target: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = pred @ target.T / float(temperature)
    labels = torch.arange(len(pred), device=pred.device)
    return F.cross_entropy(logits, labels)


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(pred, target.detach(), dim=-1)).mean()


def std_loss(features: torch.Tensor, floor: float = 0.05) -> torch.Tensor:
    std = torch.sqrt(features.float().var(dim=0, unbiased=False) + 1e-4)
    return F.relu(floor - std).mean()


@torch.no_grad()
def retrieval(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    sim = pred @ target.T
    rank = torch.argsort(sim, dim=1, descending=True)
    truth = torch.arange(len(pred), device=pred.device).unsqueeze(1)
    top1 = (rank[:, :1] == truth).any(dim=1).float().mean().item()
    top5 = (rank[:, :5] == truth).any(dim=1).float().mean().item()
    eye = torch.eye(len(pred), dtype=torch.bool, device=pred.device)
    margin = (sim.diag() - sim.masked_fill(eye, -9).max(dim=1).values).mean().item()
    return {"top1": float(top1), "top5": float(top5), "margin": float(margin)}


def make_loader(dataset: Dataset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=bool(shuffle),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def run_epoch(
    model: MaskedEvidenceJEPA,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.train(optimizer is not None)
    sums: dict[str, float] = {}
    count = 0
    preds, targets = [], []
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch)
        losses = {
            "nce": info_nce(out["pred"], out["target"], args.temperature),
            "cos": cosine_loss(out["pred"], out["target"]),
            "std": std_loss(out["pred"]) + std_loss(out["memory"]),
        }
        loss = losses["nce"] + float(args.cos_weight) * losses["cos"] + float(args.std_weight) * losses["std"]
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        bsz = int(batch["label"].shape[0])
        count += bsz
        for name, value in {"loss": loss, **losses}.items():
            sums[name] = sums.get(name, 0.0) + float(value.detach()) * bsz
        sums["pred_std"] = sums.get("pred_std", 0.0) + float(
            out["pred"].detach().std(dim=0).mean()
        ) * bsz
        preds.append(out["pred"].detach())
        targets.append(out["target"].detach())
    metrics = {name: value / max(1, count) for name, value in sums.items()}
    if preds:
        metrics.update({f"retrieval_{k}": v for k, v in retrieval(torch.cat(preds), torch.cat(targets)).items()})
    return metrics


@torch.no_grad()
def extract(
    model: MaskedEvidenceJEPA, loader: DataLoader, device: torch.device
) -> dict[str, np.ndarray]:
    model.eval()
    memory, pred, target, labels = [], [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch)
        memory.append(out["memory"].cpu().numpy())
        pred.append(out["pred"].cpu().numpy())
        target.append(out["target"].cpu().numpy())
        labels.append(batch["label"].cpu().numpy())
    return {
        "memory": np.concatenate(memory, axis=0),
        "pred": np.concatenate(pred, axis=0),
        "target": np.concatenate(target, axis=0),
        "labels": np.concatenate(labels, axis=0).astype(np.int64),
    }


def readout_metric(readout: Any, features: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    pred = readout.predict(features).astype(np.int64)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
        "confusion_matrix": confusion_matrix(
            labels, pred, labels=list(range(CLASSES))
        ).astype(int).tolist(),
        "count": int(len(labels)),
    }


def plot_diagnostic(result: dict[str, Any], out_dir: Path) -> None:
    history = result["history"]
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train"]["loss"] for row in history]
    val_loss = [row["val"]["loss"] for row in history]
    arms = ["full", "reset", "no_state"]
    bacc = [result["readout"][arm]["balanced_accuracy"] for arm in arms]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.25), dpi=180)
    fig.patch.set_facecolor("#fbfbf9")
    for ax in axes:
        ax.set_facecolor("#fbfbf9")
        ax.grid(True, axis="y", color="#d4d3cb", linewidth=0.6)
        for spine in ax.spines.values():
            spine.set_color("#333b49")
            spine.set_linewidth(0.7)
    axes[0].plot(epochs, train_loss, color="#111827", linewidth=1.5, label="train")
    axes[0].plot(epochs, val_loss, color="#d8a900", linewidth=1.5, label="val")
    axes[0].set_title("JEPA latent loss", fontsize=9, loc="left")
    axes[0].tick_params(labelsize=7, length=0)
    axes[0].legend(frameon=False, fontsize=7)
    colors = ["#111827", "#9ca3af", "#d4d3cb"]
    axes[1].bar(arms, bacc, color=colors, edgecolor="#111827", linewidth=0.5)
    axes[1].axhline(0.25, color="#7f1d1d", linestyle="--", linewidth=0.8)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("Post-hoc label readout", fontsize=9, loc="left")
    axes[1].tick_params(axis="x", labelsize=7, length=0)
    axes[1].tick_params(axis="y", labelsize=7, length=0)
    fig.suptitle(
        f"{result['env_name']} age={result['age']} seed={result['seed']}",
        fontsize=9,
        x=0.02,
        ha="left",
    )
    fig.tight_layout(pad=0.8)
    fig.savefig(out_dir / "diagnostic.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def build_datasets(args: argparse.Namespace) -> dict[str, MaskedEvidenceDataset]:
    archive = cache_path(args)
    return {
        "train_aug": MaskedEvidenceDataset(
            archive,
            age=args.age,
            split="train",
            seed=args.seed,
            validation_fraction=args.validation_fraction,
            variant="full",
            augment=True,
            temporal_drop=args.temporal_drop,
            patch_drop=args.patch_drop,
        ),
        "train_eval": MaskedEvidenceDataset(
            archive,
            age=args.age,
            split="train",
            seed=args.seed,
            validation_fraction=args.validation_fraction,
            variant="full",
            augment=False,
        ),
        "val_full": MaskedEvidenceDataset(
            archive,
            age=args.age,
            split="val",
            seed=args.seed,
            validation_fraction=args.validation_fraction,
            variant="full",
            augment=False,
        ),
        "val_reset": MaskedEvidenceDataset(
            archive,
            age=args.age,
            split="val",
            seed=args.seed,
            validation_fraction=args.validation_fraction,
            variant="reset",
            augment=False,
        ),
        "val_no_state": MaskedEvidenceDataset(
            archive,
            age=args.age,
            split="val",
            seed=args.seed,
            validation_fraction=args.validation_fraction,
            variant="no_state",
            augment=False,
        ),
    }


def train_cell(args: argparse.Namespace) -> dict[str, Any]:
    if not cache_path(args).is_file():
        raise FileNotFoundError(cache_path(args))
    out_dir = result_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(181_911 + int(args.seed) + 17 * int(args.age))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    datasets = build_datasets(args)
    train_loader = make_loader(datasets["train_aug"], args, shuffle=True)
    train_eval_loader = make_loader(datasets["train_eval"], args, shuffle=False)
    val_full_loader = make_loader(datasets["val_full"], args, shuffle=False)
    val_reset_loader = make_loader(datasets["val_reset"], args, shuffle=False)
    val_no_state_loader = make_loader(datasets["val_no_state"], args, shuffle=False)

    with np.load(cache_path(args), allow_pickle=False) as data:
        img_size = int(data["img_size"])
        action_dim = int(data["actions"].shape[-1])
    model = MaskedEvidenceJEPA(
        img_size=img_size,
        action_dim=action_dim,
        dim=int(args.dim),
        slots=int(args.slots),
        heads=int(args.heads),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
    )
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, args)
        val_metrics = run_epoch(model, val_full_loader, None, device, args)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        if epoch == 1 or epoch % max(1, int(args.epochs) // 4) == 0:
            print(
                json.dumps(
                    {
                        "env": args.env_name,
                        "age": args.age,
                        "seed": args.seed,
                        "epoch": epoch,
                        "train_loss": train_metrics["loss"],
                        "val_top1": val_metrics["retrieval_top1"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    train_features = extract(model, train_eval_loader, device)
    evals = {
        "full": extract(model, val_full_loader, device),
        "reset": extract(model, val_reset_loader, device),
        "no_state": extract(model, val_no_state_loader, device),
    }
    readout = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0))
    readout.fit(train_features["memory"], train_features["labels"])
    readout_metrics = {
        name: readout_metric(readout, payload["memory"], payload["labels"])
        for name, payload in evals.items()
    }
    retrieval_metrics = {
        name: retrieval(
            torch.from_numpy(payload["pred"]).to(device),
            torch.from_numpy(payload["target"]).to(device),
        )
        for name, payload in evals.items()
    }
    full = readout_metrics["full"]["balanced_accuracy"]
    reset = readout_metrics["reset"]["balanced_accuracy"]
    no_state = readout_metrics["no_state"]["balanced_accuracy"]
    result = {
        "schema": "masked_evidence_jepa_ogbench_cell_v1",
        "status": "completed",
        "env_name": args.env_name,
        "age": int(args.age),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "dim": int(args.dim),
        "slots": int(args.slots),
        "heads": int(args.heads),
        "training_loss_uses_cue_labels": False,
        "posthoc_readout_uses_cue_labels": True,
        "manual_cue_feature_supplied": False,
        "history": history,
        "retrieval": retrieval_metrics,
        "readout": readout_metrics,
        "gate": {
            "full_minimum": 0.75,
            "control_maximum": 0.35,
            "pass": bool(full >= 0.75 and reset <= 0.35 and no_state <= 0.35),
        },
        "claim_boundary": (
            "Autonomous masked-evidence JEPA render-stage memory probe; "
            "not a native world-model planning claim."),
    }
    (out_dir / "result.json").write_text(stable_json(result))
    np.savez_compressed(
        out_dir / "features.npz",
        train_memory=train_features["memory"],
        train_labels=train_features["labels"],
        val_full_memory=evals["full"]["memory"],
        val_reset_memory=evals["reset"]["memory"],
        val_no_state_memory=evals["no_state"]["memory"],
        val_labels=evals["full"]["labels"],
    )
    plot_diagnostic(result, out_dir)
    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "result": result,
        },
        out_dir / "model.pt",
    )
    print(stable_json({
        "env": args.env_name,
        "age": args.age,
        "seed": args.seed,
        "full": full,
        "reset": reset,
        "no_state": no_state,
        "pass": result["gate"]["pass"],
    }), flush=True)
    return result


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    for path in sorted(args.output.glob("*/*/s*/result.json")):
        rows.append(json.loads(path.read_text()))
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["env_name"], int(row["age"])), []).append(row)
    summary_rows = []
    for (env_name, age), values in sorted(grouped.items()):
        full = [v["readout"]["full"]["balanced_accuracy"] for v in values]
        reset = [v["readout"]["reset"]["balanced_accuracy"] for v in values]
        no_state = [v["readout"]["no_state"]["balanced_accuracy"] for v in values]
        top1 = [v["retrieval"]["full"]["top1"] for v in values]
        summary_rows.append({
            "env_name": env_name,
            "age": int(age),
            "seeds": [int(v["seed"]) for v in values],
            "seed_count": int(len(values)),
            "pass_count": int(sum(bool(v["gate"]["pass"]) for v in values)),
            "all_pass": bool(all(bool(v["gate"]["pass"]) for v in values)),
            "full_bacc_mean": float(np.mean(full)),
            "reset_bacc_mean": float(np.mean(reset)),
            "no_state_bacc_mean": float(np.mean(no_state)),
            "retrieval_top1_mean": float(np.mean(top1)),
        })
    summary = {
        "schema": "masked_evidence_jepa_ogbench_summary_v1",
        "status": "completed" if rows else "empty",
        "cell_count": int(len(rows)),
        "rows": summary_rows,
        "claim_boundary": (
            "Autonomous masked-evidence JEPA render-stage memory probe; "
            "labels are post-hoc evaluation only."),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(stable_json(summary))
    print(stable_json(summary), flush=True)
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    args.output = resolve_path(args.output)
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
