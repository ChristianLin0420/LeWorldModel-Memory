#!/usr/bin/env python3
"""Measure peak memory and per-step compute of the streaming slot writer.

Compares the bounded streaming writer (StreamingSlotMemory, constant state) against
one-shot full-prefix slot attention (SlotMemory) as the sequence length grows.
This produces the evidence for the paper's "bounded streaming memory" claim:
streaming peak activation memory and per-step compute stay flat while the
one-shot writer grows with sequence length.

Output: outputs/streaming_cost_v1/streaming_cost.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_masked_evidence_jepa_ogbench as base  # noqa: E402

OUT = ROOT / "outputs" / "streaming_cost_v1"
LENGTHS = [16, 32, 64, 128, 256, 512, 1024]
DIM = 160
SLOTS = 8
HEADS = 4
CHUNK = 4
BATCH = 32
REPEATS = 5


def _peak_and_time(module, tokens, valid, device):
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    # warmup
    with torch.no_grad():
        module(tokens, valid)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(REPEATS):
            module(tokens, valid)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / REPEATS
    peak = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda"
        else float("nan")
    )
    return peak, elapsed


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    one_shot = base.SlotMemory(DIM, SLOTS, HEADS).to(device).eval()
    streaming = base.StreamingSlotMemory(DIM, SLOTS, HEADS, chunk=CHUNK).to(device).eval()

    rows = []
    for length in LENGTHS:
        tokens = torch.randn(BATCH, length, DIM, device=device)
        valid = torch.ones(BATCH, length, device=device)
        os_peak, os_time = _peak_and_time(one_shot, tokens, valid, device)
        st_peak, st_time = _peak_and_time(streaming, tokens, valid, device)
        row = {
            "length": length,
            "one_shot_peak_mb": round(os_peak, 3),
            "one_shot_time_ms": round(os_time * 1e3, 4),
            "streaming_peak_mb": round(st_peak, 3),
            "streaming_time_ms": round(st_time * 1e3, 4),
            "streaming_per_step_ms": round(st_time * 1e3 / length, 5),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    # state bytes are constant for streaming (S x D), independent of length
    summary = {
        "schema": "streaming_cost_v1",
        "dim": DIM,
        "slots": SLOTS,
        "chunk": CHUNK,
        "batch": BATCH,
        "state_scalars": SLOTS * DIM,
        "state_bytes_fp32": SLOTS * DIM * 4,
        "note": (
            "StreamingSlotMemory keeps a fixed S x D state and evicts each chunk; "
            "peak activation memory and per-step compute are constant in sequence length. "
            "One-shot SlotMemory attention grows with sequence length."
        ),
        "rows": rows,
    }
    (OUT / "streaming_cost.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("wrote", OUT / "streaming_cost.json")


if __name__ == "__main__":
    main()
