#!/usr/bin/env python3
"""Temporal-coverage salient patch-set JEPA for weak OGBench rows.

This is a non-manual successor to the single-view color-target patch-set JEPA.
Instead of sampling one salient target frame, it splits the available legal
history into temporal bins and predicts salient patch sets from several bins.
The objective therefore pressures memory to carry old, middle, and recent
evidence without cue labels, cue crops, or manually declared key frames.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_masked_evidence_jepa_ogbench as base  # noqa: E402
from scripts import run_random_patchset_jepa_ogbench as patchset  # noqa: E402
from scripts import run_random_patchset_view_jepa_ogbench as view  # noqa: E402


TARGET_VIEWS = 3


class FixedPatchTargetEncoder(nn.Module):
    def __init__(self, dim: int, grid: int = 4) -> None:
        super().__init__()
        raw_dim = 7 * int(grid) * int(grid)
        self.grid = int(grid)
        generator = torch.Generator().manual_seed(1_771_903)
        projection = torch.randn(raw_dim, dim, generator=generator) / float(raw_dim) ** 0.5
        self.register_buffer("projection", projection)

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(patch, (self.grid, self.grid))
        centered = pooled - pooled.mean(dim=(2, 3), keepdim=True)
        saturation = pooled.max(dim=1, keepdim=True).values - pooled.min(dim=1, keepdim=True).values
        raw = torch.cat([pooled - 0.5, centered, saturation], dim=1).flatten(1)
        return F.normalize(raw @ self.projection, dim=-1)


class MultiViewPatchSetJEPA(patchset.RandomPatchSetJEPA):
    def __init__(self, *, img_size: int, action_dim: int, dim: int, slots: int, heads: int, chunk: int = 0) -> None:
        super().__init__(img_size=img_size, action_dim=action_dim, dim=dim, slots=slots, heads=heads, chunk=chunk)
        self.patch = FixedPatchTargetEncoder(dim)


def choose_temporal_coverage_times(
    frames: np.ndarray,
    *,
    endpoint: int,
    rng: np.random.Generator,
    variant: str,
    target_views: int = TARGET_VIEWS,
) -> list[int]:
    """Choose salient views from generic temporal bins, without labels."""
    if variant == "no_state":
        times = np.asarray(list(range(max(0, endpoint - 3), endpoint + 1)), dtype=np.int64)
    else:
        times = np.asarray(list(range(0, endpoint + 1)), dtype=np.int64)
    if len(times) == 0:
        return [0]

    selected: list[int] = []
    for chunk in np.array_split(times, max(1, int(target_views))):
        if len(chunk) == 0:
            continue
        scores = np.asarray([view.view_saliency(frames, int(time)) for time in chunk], dtype=np.float64)
        scores = scores - scores.min()
        scores = scores + 1e-6
        probs = scores / scores.sum()
        selected.append(int(rng.choice(chunk, p=probs)))

    if len(set(selected)) < target_views:
        global_scores = np.asarray([view.view_saliency(frames, int(time)) for time in times], dtype=np.float64)
        order = [int(times[i]) for i in np.argsort(global_scores)[::-1]]
        for time in order:
            if time not in selected:
                selected.append(time)
            if len(set(selected)) >= target_views:
                break
    return sorted(list(dict.fromkeys(selected)))[:target_views]


class TemporalCoveragePatchSetDataset(Dataset):
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
        cue_mode: str = "color",
    ) -> None:
        with np.load(archive, allow_pickle=False) as data:
            self.frames = data["frames"]
            self.actions = data["actions"]
            self.labels = data["cue_labels"]
            self.positions = data["cue_positions"]
        self.cue_mode = str(cue_mode)
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
        rng = np.random.default_rng(50_000_029 + episode + 431 * self.age)
        clean = self.frames[episode].copy()
        full = base.inject_cue_sequence_mode(
            clean, label, position, mode=self.cue_mode, episode=episode
        )
        source = clean if self.variant in {"reset", "no_state"} else full
        target_times = choose_temporal_coverage_times(
            source,
            endpoint=self.endpoint,
            rng=rng,
            variant=self.variant,
        )
        target_patches = []
        selected_all = []
        for target_time in target_times:
            patches, selected = view.mine_single_view_patches(source, target_time=target_time, rng=rng)
            target_patches.append(patches)
            selected_all.extend(selected)
        patches = np.concatenate(target_patches, axis=0).astype(np.uint8)

        visible = source.copy()
        if self.variant != "no_state":
            for time, y, x in selected_all:
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
    common = dict(
        age=args.age,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        cue_mode=str(getattr(args, "cue_mode", "color")),
    )
    return {
        "train_aug": TemporalCoveragePatchSetDataset(
            archive,
            split="train",
            variant="full",
            augment=True,
            temporal_drop=args.temporal_drop,
            patch_drop=args.patch_drop,
            **common,
        ),
        "train_eval": TemporalCoveragePatchSetDataset(archive, split="train", variant="full", augment=False, **common),
        "val_full": TemporalCoveragePatchSetDataset(archive, split="val", variant="full", augment=False, **common),
        "val_reset": TemporalCoveragePatchSetDataset(archive, split="val", variant="reset", augment=False, **common),
        "val_no_state": TemporalCoveragePatchSetDataset(archive, split="val", variant="no_state", augment=False, **common),
    }


_original_train_cell = patchset.train_cell


def train_cell(args):
    result = _original_train_cell(args)
    result.update(
        {
            "schema": "multiview_patchset_color_jepa_ogbench_cell_v1",
            "target_views": int(TARGET_VIEWS),
            "target_patches_per_view": int(patchset.TARGET_PATCHES),
            "target_patches": int(TARGET_VIEWS * patchset.TARGET_PATCHES),
            "claim_boundary": (
                "Temporal-coverage salient patch-set JEPA; temporal bins are generic and "
                "training still uses no cue labels, cue crops, or manually selected key frames."
            ),
        }
    )
    out_dir = base.result_dir(args)
    (out_dir / "result.json").write_text(base.stable_json(result))
    return result


patchset.RandomPatchSetJEPA = MultiViewPatchSetJEPA
patchset.build_datasets = build_datasets
patchset.train_cell = train_cell


if __name__ == "__main__":
    patchset.main()
