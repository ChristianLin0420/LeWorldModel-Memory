"""Identity-pinned loader for the released LeWM TwoRoom checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch

from lewm.models.official_lewm import OfficialLeWM
from lewm.models.official_lewm_config import (
    OfficialCheckpointIdentity,
    load_official_lewm_bundle,
)


OFFICIAL_TWOROOM_CHECKPOINT = OfficialCheckpointIdentity(
    repo_id="quentinll/lewm-tworooms",
    revision="77adaae0bc31deab21c93740d1f8bb947cd0bdec",
    config_sha256=(
        "2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09"
    ),
    weights_sha256=(
        "566f223624ea4bfb39dbfe6ae731198dd6ea73b7b8919fed6b1ecafca810f7dd"
    ),
    weights_size=72_290_849,
)


def load_official_tworoom_checkpoint(
        directory: str | Path,
        device: torch.device | str = "cpu",
        *, verify_identity: bool = True,
        ) -> OfficialLeWM:
    """Load, authenticate, and freeze an official TwoRoom bundle."""

    identity = OFFICIAL_TWOROOM_CHECKPOINT if verify_identity else None
    return load_official_lewm_bundle(directory, device, identity=identity)


__all__ = ["OFFICIAL_TWOROOM_CHECKPOINT", "load_official_tworoom_checkpoint"]
