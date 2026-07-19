#!/usr/bin/env python3
"""Single-view random patch-set JEPA for non-manual memory learning.

Compared with ``run_random_patchset_jepa_ogbench.py``, this variant avoids an
impossible target caused by masking every automatically mined evidence patch
across the stream.  It chooses one salient target frame without labels or cue
layout, mines patches inside that frame, masks only that view, and leaves other
causal observations visible.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_masked_evidence_jepa_ogbench as base  # noqa: E402
from scripts import run_random_patchset_jepa_ogbench as patchset  # noqa: E402


def view_saliency(frames: np.ndarray, time: int) -> float:
    score = patchset._saliency_map(frames, int(time))
    return float(np.percentile(score, 95) + 0.25 * score.mean())


def choose_target_time(
    frames: np.ndarray,
    *,
    endpoint: int,
    rng: np.random.Generator,
    variant: str,
) -> int:
    if variant == "no_state":
        times = list(range(max(0, endpoint - 3), endpoint + 1))
    else:
        times = list(range(0, endpoint + 1))
    scores = np.asarray([view_saliency(frames, time) for time in times], dtype=np.float64)
    scores = scores - scores.min()
    if float(scores.sum()) <= 1e-8:
        return int(rng.choice(times))
    probs = scores / scores.sum()
    return int(rng.choice(times, p=probs))


def mine_single_view_patches(
    frames: np.ndarray,
    *,
    target_time: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    size = int(patchset.PATCH_SIZE)
    h, w = frames.shape[1:3]
    stride = max(4, size // 2)
    score = patchset._saliency_map(frames, int(target_time))
    candidates: list[tuple[float, int, int, int]] = []
    for y in range(0, max(1, h - size + 1), stride):
        for x in range(0, max(1, w - size + 1), stride):
            value = patchset._patch_score(score, y, x, size)
            value += float(rng.normal(0.0, 1e-3))
            candidates.append((value, int(target_time), int(y), int(x)))
    candidates.sort(reverse=True, key=lambda row: row[0])
    selected: list[tuple[int, int, int]] = []
    for _, time, y, x in candidates:
        candidate = (time, y, x)
        if patchset._too_close(candidate, selected):
            continue
        selected.append(candidate)
        if len(selected) >= patchset.TARGET_PATCHES:
            break
    while len(selected) < patchset.TARGET_PATCHES:
        selected.append(
            (
                int(target_time),
                int(rng.integers(0, max(1, h - size + 1))),
                int(rng.integers(0, max(1, w - size + 1))),
            )
        )
    patches = np.stack([patchset._safe_crop(frames[t], y, x, size) for t, y, x in selected]).astype(np.uint8)
    return patches, selected


class SingleViewPatchSetDataset(Dataset):
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
        self.endpoint = base.LAST_CUE_FRAME + self.age
        self.variant = str(variant)
        self.augment = bool(augment)
        self.temporal_drop = float(temporal_drop)
        self.patch_drop = float(patch_drop)
        rng = np.random.default_rng(97_101 + int(seed))
        order = rng.permutation(len(self.frames))
        val_count = max(base.CLASSES, int(round(len(order) * float(validation_fraction))))
        self.indices = order[:-val_count] if split == "train" else order[-val_count:]
        if self.endpoint >= self.frames.shape[1]:
            raise ValueError(f"age {age} exceeds cached sequence length")

    def __len__(self) -> int:
        return int(len(self.indices))

    def _valid_times(self, rng: np.random.Generator) -> list[int]:
        if self.variant == "no_state":
            return list(range(max(0, self.endpoint - 3), self.endpoint + 1))
        times = list(range(0, self.endpoint + 1))
        if self.augment and self.temporal_drop > 0:
            times = [time for time in times if time == self.endpoint or rng.random() > self.temporal_drop]
            if not times:
                times = [self.endpoint]
        return sorted(set(times))

    def _maybe_mask_random_patch(self, frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if not self.augment or rng.random() >= self.patch_drop:
            return frame
        out = frame.copy()
        size = max(6, frame.shape[0] // 5)
        x = int(rng.integers(0, max(1, frame.shape[1] - size)))
        y = int(rng.integers(0, max(1, frame.shape[0] - size)))
        patchset._mask_patch(out, y, x, size)
        return out

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        episode = int(self.indices[item])
        label = int(self.labels[episode])
        position = int(self.positions[episode])
        rng = np.random.default_rng(40_000_019 + episode + 397 * self.age)
        clean = self.frames[episode].copy()
        full = base.inject_cue_sequence(clean, label, position)
        source = clean if self.variant in {"reset", "no_state"} else full
        target_time = choose_target_time(source, endpoint=self.endpoint, rng=rng, variant=self.variant)
        patches, selected = mine_single_view_patches(source, target_time=target_time, rng=rng)
        visible = source.copy()
        if self.variant != "no_state":
            for time, y, x in selected:
                patchset._mask_patch(visible[int(time)], int(y), int(x), patchset.PATCH_SIZE)
        times = self._valid_times(rng)
        frame_tokens = np.zeros(
            (base.MAX_CONTEXT, clean.shape[-3], clean.shape[-2], clean.shape[-1]),
            dtype=np.uint8,
        )
        action_tokens = np.zeros((base.MAX_CONTEXT, self.actions.shape[-1]), dtype=np.float32)
        time_tokens = np.zeros((base.MAX_CONTEXT, 1), dtype=np.float32)
        valid = np.zeros((base.MAX_CONTEXT,), dtype=np.float32)
        for slot, time in enumerate(times[:base.MAX_CONTEXT]):
            frame_tokens[slot] = self._maybe_mask_random_patch(visible[time], rng)
            if time > 0:
                action_tokens[slot] = self.actions[episode, time - 1]
            time_tokens[slot, 0] = float(time) / float(base.LENGTH - 1)
            valid[slot] = 1.0
        return {
            "frames": torch.from_numpy(frame_tokens.astype(np.float32) / 255.0).permute(0, 3, 1, 2),
            "actions": torch.from_numpy(action_tokens),
            "times": torch.from_numpy(time_tokens),
            "valid": torch.from_numpy(valid),
            "target_patches": torch.from_numpy(patches.astype(np.float32) / 255.0).permute(0, 3, 1, 2),
            "label": torch.tensor(label, dtype=torch.long),
        }


def build_datasets(args):
    archive = base.cache_path(args)
    common = dict(age=args.age, seed=args.seed, validation_fraction=args.validation_fraction)
    return {
        "train_aug": SingleViewPatchSetDataset(
            archive,
            split="train",
            variant="full",
            augment=True,
            temporal_drop=args.temporal_drop,
            patch_drop=args.patch_drop,
            **common,
        ),
        "train_eval": SingleViewPatchSetDataset(archive, split="train", variant="full", augment=False, **common),
        "val_full": SingleViewPatchSetDataset(archive, split="val", variant="full", augment=False, **common),
        "val_reset": SingleViewPatchSetDataset(archive, split="val", variant="reset", augment=False, **common),
        "val_no_state": SingleViewPatchSetDataset(archive, split="val", variant="no_state", augment=False, **common),
    }


patchset.build_datasets = build_datasets


if __name__ == "__main__":
    patchset.main()
