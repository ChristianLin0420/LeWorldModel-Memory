#!/usr/bin/env python3
"""Single-view salient patch-set JEPA with fixed generic patch targets.

This keeps the non-manual v2 data flow:
  * choose a salient target frame without cue labels or cue-layout knowledge;
  * mine target patches without cue crops or key-frame annotations;
  * mask only the selected target view while leaving redundant causal evidence.

The only change is the target latent.  Instead of a frozen random CNN on small
patches, this uses a fixed generic color/texture projection, matching the
stable target style that worked in the fixed-panel diagnostic stage.
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_random_patchset_jepa_ogbench as patchset  # noqa: E402
from scripts import run_random_patchset_view_jepa_ogbench as view  # noqa: E402


class FixedPatchTargetEncoder(nn.Module):
    def __init__(self, dim: int, grid: int = 4) -> None:
        super().__init__()
        raw_dim = 7 * int(grid) * int(grid)
        self.grid = int(grid)
        projection = torch.randn(raw_dim, dim) / float(raw_dim) ** 0.5
        self.register_buffer("projection", projection)

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(patch, (self.grid, self.grid))
        centered = pooled - pooled.mean(dim=(2, 3), keepdim=True)
        saturation = pooled.max(dim=1, keepdim=True).values - pooled.min(dim=1, keepdim=True).values
        raw = torch.cat([pooled - 0.5, centered, saturation], dim=1).flatten(1)
        return F.normalize(raw @ self.projection, dim=-1)


class ColorTargetPatchSetJEPA(patchset.RandomPatchSetJEPA):
    def __init__(self, *, img_size: int, action_dim: int, dim: int, slots: int, heads: int) -> None:
        super().__init__(img_size=img_size, action_dim=action_dim, dim=dim, slots=slots, heads=heads)
        self.patch = FixedPatchTargetEncoder(dim)


patchset.RandomPatchSetJEPA = ColorTargetPatchSetJEPA
patchset.build_datasets = view.build_datasets


if __name__ == "__main__":
    patchset.main()
