#!/usr/bin/env python3
"""V21 X3 — the salience threshold s* on a second host (docs/V21_PROPOSAL.md
4/X3): frozen pretrained DINOv2 ViT-S features over the W0 salience ladder.

The W0 instrument measured s*(vicreg) = t1s2 (a single-pixel border tint)
on the one healthy trained host.  To make s* an INSTRUMENT rather than a
single-host observation (panel objection I5), this script repeats the
sighted-certificate readout on a second, architecturally unrelated encoder:
the frozen DINOv2 backbone (vit_small_patch14_dinov2, 384-d features, no
training anywhere — general-purpose pretrained representation).

Per ladder level (t1s1 < t1s2 < t1s3 < t1) x 3 bank seeds: the P1b sighted
probe (8 frames spanning the cue window + episode + mean-pool, logistic,
pass >= 0.75), level pass = majority.  s*(dino) = lowest passing level.
Either outcome informs: s*(dino) < s*(vicreg) => the JEPA training regime
raised the threshold (deletion pressure); equal/higher => the threshold is
representation-general at this resolution.

Writes outputs/v21_x3/dino_sstar.{json,md}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.tasks_v19 import make_task
import scripts.certify_v19_p1b as p1b

X3 = ROOT / "outputs" / "v21_x3"
LADDER = ("t1s1", "t1s2", "t1s3", "t1")
SEEDS = (0, 1, 2)
ENCODE_BATCH = 256


class DinoFeatures:
    def __init__(self, device: torch.device):
        import timm
        self.model = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m", pretrained=True,
            num_classes=0, img_size=224, dynamic_img_size=True
        ).to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.mean = torch.tensor([0.485, 0.456, 0.406],
                                 device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225],
                                device=device).view(1, 3, 1, 1)
        self.device = device

    @torch.no_grad()
    def encode_bank(self, bank) -> np.ndarray:
        """(E, L, 384) frozen DINOv2 features of uint8 frames."""
        episodes, length = bank.num_episodes, bank.length
        flat = torch.from_numpy(
            bank.frames.reshape(-1, 64, 64, 3).astype(np.float32) / 255.0
        ).permute(0, 3, 1, 2)
        outputs = []
        for start in range(0, len(flat), ENCODE_BATCH):
            chunk = flat[start:start + ENCODE_BATCH].to(self.device)
            chunk = torch.nn.functional.interpolate(
                chunk, size=(224, 224), mode="bilinear", align_corners=False)
            chunk = (chunk - self.mean) / self.std
            outputs.append(self.model(chunk).float().cpu().numpy())
        return np.concatenate(outputs).reshape(episodes, length, -1)


def certify_level(encoder: DinoFeatures, level: str, seed: int
                  ) -> dict[str, Any]:
    task = make_task(level)
    train_seed, eval_seed = p1b.bank_seeds(seed)
    train_bank = task.generate(p1b.STREAM, p1b.DEFAULT_E_TRAIN, train_seed)
    eval_bank = task.generate(p1b.STREAM, p1b.DEFAULT_E_EVAL, eval_seed)
    emb_train = encoder.encode_bank(train_bank)
    emb_eval = encoder.encode_bank(eval_bank)
    fit = p1b._probe_fn("cat")
    score = float(fit(
        p1b.sighted_features_cat(emb_train, train_bank), train_bank.xi,
        p1b.sighted_features_cat(emb_eval, eval_bank), eval_bank.xi))
    return {"level": level, "seed": seed, "sighted_score": round(score, 4),
            "pass": bool(score >= p1b.SIGHTED_ACC_MIN)}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--levels", default=",".join(LADDER),
                        help="ascending-salience ladder to certify")
    parser.add_argument("--stem", default="dino_sstar",
                        help="output filename stem under outputs/v21_x3")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    ladder = tuple(args.levels.split(","))
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available()
        else "cpu")
    encoder = DinoFeatures(device)
    levels: dict[str, Any] = {}
    s_star = None
    for level in ladder:
        cells = [certify_level(encoder, level, seed) for seed in SEEDS]
        passes = sum(cell["pass"] for cell in cells)
        level_pass = passes >= 2
        levels[level] = {
            "scores": [cell["sighted_score"] for cell in cells],
            "passes": f"{passes}/{len(SEEDS)}",
            "level_pass": bool(level_pass),
        }
        print(f"[v21-x3-dino] {level}: {levels[level]['scores']} "
              f"({'PASS' if level_pass else 'fail'})", flush=True)
        if level_pass and s_star is None:
            s_star = level
    report = {
        "schema_version": 1,
        "study": "v21-x3-dino-salience-threshold",
        "encoder": "vit_small_patch14_dinov2.lvd142m (frozen, 384-d)",
        "gate": p1b.SIGHTED_ACC_MIN,
        "ladder": list(ladder),
        "levels": levels,
        "s_star_dino": s_star,
        "s_star_vicreg_reference": "t1s2 (outputs/v20_w0/w0_summary.md)",
    }
    X3.mkdir(parents=True, exist_ok=True)
    (X3 / f"{args.stem}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = ["# V21 X3 — s* on frozen DINOv2 (second host)", "",
             "| level | sighted scores (3 seeds) | pass |", "|---|---|---|"]
    for level, row in levels.items():
        lines.append(f"| {level} | {row['scores']} | "
                     f"{'PASS' if row['level_pass'] else 'fail'} |")
    lines.append("")
    lines.append(f"**s\\*(dino)** = `{s_star}` · s\\*(vicreg) = `t1s2`")
    (X3 / f"{args.stem}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
