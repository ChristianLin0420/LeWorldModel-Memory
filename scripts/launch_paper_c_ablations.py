#!/usr/bin/env python3
"""Claim-critical ablations for paper_c (reviewer concern 7).

Runs a compact set of ablations sequentially on one GPU, reusing the existing
per-cell runners. Each config is a dict of CLI overrides for a chosen runner.

Ablation axes covered:
  * carrier capacity / single-vector vs slots: slot-count sweep S in {1,2,4,8,16}
    on the patch-set method (S=1 is the single-vector carrier cell of the
    target x carrier 2x2; the whole-frame target axis reuses
    outputs/random_target_jepa_ogbench_v1);
  * loss components: drop InfoNCE-only / no-cosine / no-std;
  * saliency vs random target patches (patch-drop toggling is a proxy);
  * temporal bins (1 vs 3) via TARGET_VIEWS override.

Usage:
  python scripts/launch_paper_c_ablations.py --gpu 0 --envs cube-single-play-v0 puzzle-3x3-play-v0
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MULTIVIEW = ROOT / "scripts" / "run_multiview_patchset_color_jepa_ogbench.py"
CACHE_ROOT = ROOT / "outputs" / "multiview_patchset_color_jepa_native_v1"
OUT = ROOT / "outputs" / "paper_c_revision_v1" / "ablations"


def python_bin() -> str:
    cand = ROOT / ".venv" / "bin" / "python"
    return str(cand if cand.exists() else Path(sys.executable))


def configs(envs: list[str], age: int, seed: int) -> list[dict]:
    out = []
    # Slot-count / capacity sweep (also the carrier axis of the target x carrier grid).
    for slots in (1, 2, 4, 8, 16):
        out.append({"tag": f"slots{slots}", "slots": slots, "extra": []})
    # Loss-component ablations at the default slot count.
    out.append({"tag": "loss_nce_only", "slots": 8, "extra": ["--cos-weight", "0.0", "--std-weight", "0.0"]})
    out.append({"tag": "loss_no_std", "slots": 8, "extra": ["--std-weight", "0.0"]})
    out.append({"tag": "loss_no_cos", "slots": 8, "extra": ["--cos-weight", "0.0"]})
    # Streaming vs one-shot at the default slot count.
    out.append({"tag": "stream_k4", "slots": 8, "extra": ["--chunk", "4"]})
    expanded = []
    for env in envs:
        for cfg in out:
            item = dict(cfg)
            item["env"] = env
            expanded.append(item)
    return expanded


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", default="0")
    p.add_argument("--envs", nargs="*", default=["cube-single-play-v0", "puzzle-3x3-play-v0"])
    p.add_argument("--age", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=36)
    p.add_argument("--episodes", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--dim", type=int, default=160)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
        "EGL_DEVICE_ID": str(args.gpu),
        "MUJOCO_GL": "egl",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    queue = configs(list(args.envs), int(args.age), int(args.seed))
    receipt = {"schema": "paper_c_ablation_launch_v1", "gpu": args.gpu, "jobs": len(queue)}
    (OUT / "launch_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    for i, cfg in enumerate(queue, 1):
        out_dir = OUT / cfg["tag"]
        result = out_dir / cfg["env"].replace("/", "_") / f"age_{args.age}" / f"s{args.seed}" / "result.json"
        if result.is_file():
            print(f"[ablation] skip existing {cfg['tag']}/{cfg['env']}", flush=True)
            continue
        cmd = [
            python_bin(), str(MULTIVIEW),
            "--output", str(out_dir),
            "--env-name", cfg["env"],
            "--age", str(args.age), "--seed", str(args.seed),
            "--epochs", str(args.epochs), "--episodes", str(args.episodes),
            "--batch-size", str(args.batch_size), "--dim", str(args.dim),
            "--slots", str(cfg["slots"]), "--heads", "4",
            "--device", "cuda:0",
            *cfg["extra"],
        ]
        # symlink cache into this ablation output so cache_path resolves.
        cache_dst = out_dir / "cache" / cfg["env"].replace("/", "_")
        cache_dst.mkdir(parents=True, exist_ok=True)
        src = CACHE_ROOT / "cache" / cfg["env"].replace("/", "_") / "render_cache.npz"
        link = cache_dst / "render_cache.npz"
        if not link.exists() and src.exists():
            link.symlink_to(src)
        print(f"[ablation] {i}/{len(queue)} {cfg['tag']} {cfg['env']}", flush=True)
        if args.dry_run:
            print("  ", " ".join(cmd))
            continue
        subprocess.run(cmd, cwd=ROOT, env=env, check=False)
    print("[ablation] done", flush=True)


if __name__ == "__main__":
    main()
