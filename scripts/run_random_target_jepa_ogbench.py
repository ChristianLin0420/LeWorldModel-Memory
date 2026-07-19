#!/usr/bin/env python3
"""Random-target Masked-Evidence JEPA variant.

This variant removes the hand-designed evidence panel from
``run_masked_evidence_jepa_ogbench.py``.  For each trajectory it samples a
target frame from the causal past, withholds that frame from the visible
history, and predicts a stop-gradient latent of the whole target frame.

Cue labels are still used only for post-hoc evaluation.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_masked_evidence_jepa_ogbench as base  # noqa: E402


class RandomTargetDataset(base.MaskedEvidenceDataset):
    """Same OGBench cache, but no fixed evidence panel or key-frame target."""

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        episode = int(self.indices[item])
        label = int(self.labels[episode])
        position = int(self.positions[episode])
        rng = np.random.default_rng(20_000_003 + episode + 997 * self.age)
        clean = self.frames[episode].copy()
        full = base.inject_cue_sequence(clean, label, position)
        source = clean if self.variant in {"reset", "no_state"} else full

        if self.variant == "no_state":
            target_time = int(rng.integers(max(0, self.endpoint - 3), self.endpoint + 1))
        else:
            target_time = int(rng.integers(0, self.endpoint + 1))

        times = self._valid_times(rng)
        times = [time for time in times if time != target_time]
        if not times:
            times = [self.endpoint]
        frame_tokens = np.zeros(
            (base.MAX_CONTEXT, clean.shape[-3], clean.shape[-2], clean.shape[-1]),
            dtype=np.uint8,
        )
        action_tokens = np.zeros((base.MAX_CONTEXT, self.actions.shape[-1]), dtype=np.float32)
        time_tokens = np.zeros((base.MAX_CONTEXT, 1), dtype=np.float32)
        valid = np.zeros((base.MAX_CONTEXT,), dtype=np.float32)
        for slot, time in enumerate(times[:base.MAX_CONTEXT]):
            frame_tokens[slot] = self._maybe_mask_patch(source[time], rng)
            if time > 0:
                action_tokens[slot] = self.actions[episode, time - 1]
            time_tokens[slot, 0] = float(time) / float(base.LENGTH - 1)
            valid[slot] = 1.0

        return {
            "frames": torch.from_numpy(frame_tokens.astype(np.float32) / 255.0).permute(0, 3, 1, 2),
            "actions": torch.from_numpy(action_tokens),
            "times": torch.from_numpy(time_tokens),
            "valid": torch.from_numpy(valid),
            "target_panel": torch.from_numpy(source[target_time].astype(np.float32) / 255.0).permute(2, 0, 1),
            "label": torch.tensor(label, dtype=torch.long),
        }


def build_datasets(args):
    archive = base.cache_path(args)
    return {
        "train_aug": RandomTargetDataset(
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
        "train_eval": RandomTargetDataset(
            archive,
            age=args.age,
            split="train",
            seed=args.seed,
            validation_fraction=args.validation_fraction,
            variant="full",
            augment=False,
        ),
        "val_full": RandomTargetDataset(
            archive,
            age=args.age,
            split="val",
            seed=args.seed,
            validation_fraction=args.validation_fraction,
            variant="full",
            augment=False,
        ),
        "val_reset": RandomTargetDataset(
            archive,
            age=args.age,
            split="val",
            seed=args.seed,
            validation_fraction=args.validation_fraction,
            variant="reset",
            augment=False,
        ),
        "val_no_state": RandomTargetDataset(
            archive,
            age=args.age,
            split="val",
            seed=args.seed,
            validation_fraction=args.validation_fraction,
            variant="no_state",
            augment=False,
        ),
    }


base.build_datasets = build_datasets


if __name__ == "__main__":
    base.main()
