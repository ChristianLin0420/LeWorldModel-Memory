#!/usr/bin/env python3
"""V20 W1 deployment evaluation: the DFC variants + the probe battery
(docs/V20_PROPOSAL.md 4.3/4.5).

For every trained ``lkc_rfix`` checkpoint under the W1 root this script runs
the deployment-time slow filter (lewm/models/v20_dfc.py) over the frozen val
bank in streaming order and writes one eval_export.npz per variant:

    dfc_rho6 / dfc_rho4 / dfc_rho2    derived gain, rho = 1e-6 / 1e-4 / 1e-2
    dfc_eta3 / dfc_eta2 / dfc_eta1    AdaJEPA-style fixed eta = 1e-3/1e-2/1e-1

(the plain ``lkc_rfix`` export written by the trainer IS the rho = 0 limit —
the subsumption reference).  Exports land beside the trained arms in the P2
layout, so scripts/eval_v19_p2.py's probe battery scores every arm and
variant identically; this script finishes by running those probes over the
whole root.
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

from lewm.models.v19_carriers import LatentKalmanCell, make_carrier
from lewm.models.v20_dfc import SlowFilterConfig, dfc_stream_eval
from lewm.tasks_v19.base import EpisodeBatch, load_bank
import scripts.certify_v19_p1b as p1b
import scripts.eval_v19_p2 as p2eval
import scripts.train_v19_p0 as p0
import scripts.train_v19_p2 as p2
import scripts.train_v20_w1 as w1

DFC_VARIANTS: dict[str, SlowFilterConfig] = {
    "dfc_rho6": SlowFilterConfig(rho=1e-6),
    "dfc_rho4": SlowFilterConfig(rho=1e-4),
    "dfc_rho2": SlowFilterConfig(rho=1e-2),
    "dfc_eta3": SlowFilterConfig(rho=0.0, eta_fixed=1e-3),
    "dfc_eta2": SlowFilterConfig(rho=0.0, eta_fixed=1e-2),
    "dfc_eta1": SlowFilterConfig(rho=0.0, eta_fixed=1e-1),
}


def load_rfix_checkpoint(path: Path, device: torch.device
                         ) -> tuple[torch.nn.Module, LatentKalmanCell, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("arm") != "lkc_rfix":
        raise ValueError(f"{path} is not an lkc_rfix checkpoint")
    host = checkpoint["host"]
    action_dim = int(checkpoint["action_dim"])
    model = w1.build_host(host, action_dim)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    embed_dim = int(p0.HOST_CONFIGS[
        "vicreg" if host == "vicreg" else "sigreg"]["embed_dim"])
    carrier = make_carrier("lkc_rfix", embed_dim, action_dim)
    carrier.load_state_dict(checkpoint["carrier_state_dict"], strict=True)
    model.to(device).eval()
    carrier.to(device).eval()
    for module in (model, carrier):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return model, carrier, checkpoint


def encode_val(host: str, model: torch.nn.Module, bank: EpisodeBatch,
               device: torch.device) -> torch.Tensor:
    """(E, L, D) frozen fp32 embeddings via the P1b chunked encoder."""
    embeddings = p1b.encode_bank(w1.encode_host(host), model, bank, device)
    return torch.from_numpy(embeddings).to(device)


def export_variant(variant: str, config: SlowFilterConfig,
                   carrier: LatentKalmanCell, z: torch.Tensor,
                   bank: EpisodeBatch, checkpoint: dict, out_path: Path
                   ) -> None:
    actions = torch.from_numpy(bank.actions.astype(np.float32)).to(z.device)
    result = dfc_stream_eval(carrier, z, actions, config)
    arrays: dict[str, np.ndarray] = {
        "prior_read": result.prior_read,
        "enc_o0": z[:, 0].cpu().numpy().astype(np.float32),
        "actions": bank.actions.astype(np.float32),
        "xi": bank.xi,
    }
    for name, value in bank.events.items():
        arrays[f"event_{name}"] = value
    for key, value in result.telemetry.items():
        arrays[f"tel_{key}"] = value
    for key, value in result.phi_trace.items():
        arrays[f"phi_{key}"] = value
    if bank.xi_kind == "cont":
        # The continuous-task probe references (p2.export_eval convention).
        from lewm.tasks_v19 import make_task
        task = make_task(checkpoint["task"])
        index = np.arange(bank.num_episodes)
        gap_on = bank.events["gap_on"]
        arrays["posterior_mean"] = task.posterior_mean_prediction(
            bank).astype(np.float32)
        arrays["frozen_pos"] = task._normalize(
            bank.exo_state[index, gap_on - 1, 0:2]).astype(np.float32)
    meta = {
        "schema_version": p2.EXPORT_SCHEMA_VERSION,
        "task": checkpoint["task"],
        "arm": variant,
        "seed": checkpoint["seed"],
        "host": checkpoint["host"],
        "xi_kind": bank.xi_kind,
        "n_classes": bank.n_classes,
        "episodes": bank.num_episodes,
        "length": bank.length,
        "embed_dim": int(arrays["enc_o0"].shape[-1]),
        "carrier": carrier.describe(),
        "dfc_config": result.config,
        "base_arm": "lkc_rfix",
        "stream": bank.stream,
    }
    p2.write_eval_export(out_path, arrays, meta)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/v20_w1")
    parser.add_argument("--tasks", default="t1dev,t3dev")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--p2-data-root", default="outputs/v19_p2/data")
    parser.add_argument("--variants", default=",".join(DFC_VARIANTS))
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available()
        else "cpu")
    root = Path(args.root)
    tasks = [name.strip() for name in args.tasks.split(",") if name.strip()]
    seeds = [int(value) for value in args.seeds.split(",")]
    variants = [name.strip() for name in args.variants.split(",")
                if name.strip()]
    for variant in variants:
        if variant not in DFC_VARIANTS:
            raise ValueError(f"unknown DFC variant {variant!r}")

    for task in tasks:
        paths, _ = w1.resolve_banks(task, args.p2_data_root, root / "data")
        val_bank = load_bank(paths["val"]["observed"])
        for seed in seeds:
            checkpoint_path = (root / task / "lkc_rfix" / f"s{seed}"
                               / "checkpoint.pt")
            if not checkpoint_path.exists():
                print(f"[v20-w1-eval] MISSING {checkpoint_path} — skipped",
                      flush=True)
                continue
            pending = [variant for variant in variants
                       if not (root / task / variant / f"s{seed}"
                               / "eval_export.npz").exists()]
            if not pending:
                continue
            model, carrier, checkpoint = load_rfix_checkpoint(
                checkpoint_path, device)
            z = encode_val(checkpoint["host"], model, val_bank, device)
            for variant in pending:
                out_path = (root / task / variant / f"s{seed}"
                            / "eval_export.npz")
                export_variant(variant, DFC_VARIANTS[variant], carrier, z,
                               val_bank, checkpoint, out_path)
                print(f"[v20-w1-eval] wrote {out_path}", flush=True)
            del model, carrier, z
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Probe battery over every export under the root (trained arms + DFC
    # variants) — the V19 machinery verbatim.
    exports = p2eval.discover_exports(root)
    print(f"[v20-w1-eval] probing {len(exports)} exports", flush=True)
    for export_path in exports:
        results_path = export_path.parent / p2eval.RESULTS_NAME
        if results_path.exists():
            continue
        summary = p2eval.process_run(export_path)
        print(f"[v20-w1-eval] probes {export_path.parent}: "
              f"registered={summary['registered']['mean']:.3f}", flush=True)


if __name__ == "__main__":
    main()
