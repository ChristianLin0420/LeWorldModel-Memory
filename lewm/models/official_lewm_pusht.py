"""Identity-pinned loader for the official LeWM PushT checkpoint.

The values below come from the official ``quentinll/lewm-pusht`` model repo at
revision ``22b330c``.  The LFS object id published by Hugging Face is the
SHA-256 of ``weights.pt``.  Pinning it is necessary because PushT and Reacher
currently share an architecture, so state-dict strictness alone cannot detect
an environment-swapped checkpoint.

Model repository: https://huggingface.co/quentinll/lewm-pusht
Official collection: https://huggingface.co/collections/quentinll/lewm
"""

from __future__ import annotations

from pathlib import Path

import torch

from lewm.models.official_lewm import OfficialLeWM
from lewm.models.official_lewm_config import (
    OfficialCheckpointIdentity,
    load_official_lewm_bundle,
)


OFFICIAL_PUSHT_CHECKPOINT = OfficialCheckpointIdentity(
    repo_id="quentinll/lewm-pusht",
    revision="22b330c28c27ead4bfd1888615af1340e3fe9052",
    config_sha256=(
        "2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09"
    ),
    weights_sha256=(
        "48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607"
    ),
    weights_size=72_290_721,
)


def load_official_pusht_checkpoint(
        directory: str | Path,
        device: torch.device | str = "cpu",
        *, verify_identity: bool = True,
        ) -> OfficialLeWM:
    """Load an already-downloaded official PushT bundle.

    ``directory`` must contain the official ``config.json`` and ``weights.pt``.
    This function intentionally performs no network access.  Identity checking
    should only be disabled for explicit checkpoint-conversion diagnostics.
    State-dict keys, shapes, and dtypes remain strict in either mode.
    """
    identity = OFFICIAL_PUSHT_CHECKPOINT if verify_identity else None
    return load_official_lewm_bundle(directory, device, identity=identity)

