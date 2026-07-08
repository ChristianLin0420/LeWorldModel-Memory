import sys
from pathlib import Path

import torch
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.official_lewm import preprocess_frames


def test_preprocess_matches_released_lewm_transform_exactly() -> None:
    generator = torch.Generator().manual_seed(20260706)
    frames = torch.randint(
        0, 256, (2, 3, 64, 64), dtype=torch.uint8, generator=generator)
    reference = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        v2.Resize(size=224),
    ])(frames)

    actual = preprocess_frames(frames)

    assert torch.equal(actual, reference)
