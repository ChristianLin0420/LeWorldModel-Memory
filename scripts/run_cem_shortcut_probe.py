#!/usr/bin/env python3
"""Head-to-head: label-free surprise WRITE gate vs the old saliency miner on the
colour-shortcut task (``shape_random_color``).

Mechanistic claim under test (docs/EMERGENT_CUE_MEMORY_PARADIGM.md sec.C):

    "A colour the host already predicts has ZERO reducible surprise and is never
     written."

So the CEM surprise gate (= a FROZEN host's own one-step prediction error) should
write the *identity-bearing but unpredictable* cue, while the hand-authored
saliency miner (a colour/brightness heuristic) is captured by a salient-but-
predictable distractor and fails to retain identity.

Setup (self-contained; reuses the shape/colour cue renderers from
``run_masked_evidence_jepa_ogbench.py``):
  * A predictable moving-gradient background.
  * The CUE on frames 1..3: a SHAPE (=label, 4 classes) in a RANDOM colour
    (``shape_random_color``) -- unpredictable, colour-decorrelated from label.
  * A DISTRACTOR on a later frame: a bright, FIXED-colour patch present in EVERY
    episode -- maximally salient by colour, but label-independent and, crucially,
    PREDICTABLE (the frozen host was trained on backgrounds that contain it).
  * A frozen mini-host (tiny CNN encoder + linear next-latent predictor) trained
    only on background+distractor sequences (no cue), then frozen.  Surprise is
    its one-step latent error -- label-free.

WRITE policies (budget = 1 frame per episode):
  * surprise : argmax_t host-surprise(t)         (CEM, label-free)
  * saliency : argmax_t colour-saliency(t)        (old miner, colour heuristic)

Metric: post-hoc linear readout of the LABEL from the frozen-encoder features of
the WRITTEN frame.  A writer that retains identity gives high balanced accuracy;
one captured by the colour distractor sits at chance (0.25).

Also runs the plain ``color`` cue (label == colour) as a control.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_masked_evidence_jepa_ogbench import (  # noqa: E402
    PALETTE, _RANDOM_PALETTE, draw_cue, draw_cue_shape,
)

CLASSES = 4
IMG = 48
LENGTH = 12
CUE_FRAMES = (1, 2, 3)
DISTRACTOR_FRAME = 7
DISTRACTOR_COLOR = np.asarray([255, 40, 40], dtype=np.uint8)  # bright, fixed, salient
DEFAULT_OUTPUT = ROOT / "outputs/cem_shortcut_v1"
DEFAULT_DINOV2 = ROOT / "outputs/dinowm_native_pusht_audit_v1/vendor/dinov2"
DEFAULT_TORCH_HOME = ROOT / "outputs/dinowm_native_pusht_audit_v1/torch_home"


def background(episode: int) -> np.ndarray:
    """A static (temporally predictable) gradient background.

    The background is constant across time within an episode (only its spatial
    pattern varies across episodes), so a frozen next-frame host trivially
    predicts it -> reducible surprise ~0 on background frames.  The only
    unpredictable events are the transient cue (novel) and, before the host is
    trained on it, the distractor (which is made predictable via training).
    """
    rng = np.random.default_rng(4242 + episode)
    phase = rng.uniform(0, np.pi)
    yy, xx = np.mgrid[0:IMG, 0:IMG]
    val = 90 + 40 * np.sin(0.15 * xx + phase) + 30 * np.cos(0.15 * yy - phase)
    base = np.clip(val, 30, 200).astype(np.uint8)
    frame = np.stack([base, base, base], axis=-1)
    return np.repeat(frame[None], LENGTH, axis=0)


def add_distractor(frames: np.ndarray) -> np.ndarray:
    """A bright fixed-colour patch (predictable, salient, label-independent)."""
    out = frames.copy()
    x0, y0, s = IMG - 16, IMG - 16, 12
    out[DISTRACTOR_FRAME, y0:y0 + s, x0:x0 + s] = DISTRACTOR_COLOR
    return out


def render_episode(episode: int, label: int, mode: str) -> np.ndarray:
    frames = add_distractor(background(episode))
    rng = np.random.default_rng(9001 + episode)
    position = 0
    if mode == "shape_random_color":
        color = _RANDOM_PALETTE[int(rng.integers(0, len(_RANDOM_PALETTE)))]
        for t in CUE_FRAMES:
            frames[t] = draw_cue_shape(frames[t], int(label), position, color)
    elif mode == "color":
        for t in CUE_FRAMES:
            frames[t] = draw_cue(frames[t], int(label), position)
    else:
        raise ValueError(mode)
    return frames


class MiniHost(nn.Module):
    """Tiny encoder + linear next-latent predictor (the frozen surprise host)."""

    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 48, 3, 2, 1), nn.GroupNorm(8, 48), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(48, dim))
        self.pred = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)

    def forward_pred(self, z: torch.Tensor) -> torch.Tensor:
        return self.pred(z)


class FrozenSemanticEncoder:
    """Locally cached frozen DINOv2 patch-token target (stop-gradient)."""
    def __init__(self, device, repo: Path, torch_home: Path):
        os.environ["TORCH_HOME"] = str(torch_home.resolve())
        self.model = torch.hub.load(
            str(repo.resolve()), "dinov2_vits14", source="local",
            pretrained=True).eval().to(device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def patches(self, frames_t: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            frames_t, size=(224, 224), mode="bilinear", align_corners=False)
        mean = torch.tensor(
            [0.485, 0.456, 0.406], device=x.device)[None, :, None, None]
        std = torch.tensor(
            [0.229, 0.224, 0.225], device=x.device)[None, :, None, None]
        return self.model.forward_features(
            (x - mean) / std)["x_norm_patchtokens"].float()


def to_tensor(frames: np.ndarray, device) -> torch.Tensor:
    x = torch.from_numpy(frames.astype(np.float32) / 255.0)
    return x.permute(0, 3, 1, 2).to(device)


def train_frozen_host(device, n_bg: int = 256, steps: int = 400) -> MiniHost:
    """Train the mini-host on background+distractor (NO cue), then freeze it."""
    host = MiniHost().to(device)
    opt = torch.optim.AdamW(host.parameters(), lr=2e-3)
    seqs = [add_distractor(background(10_000 + i)) for i in range(n_bg)]
    seqs = torch.stack([to_tensor(s, device) for s in seqs])  # (N,L,3,H,W)
    host.train()
    rng = np.random.default_rng(0)
    for step in range(steps):
        idx = rng.integers(0, n_bg, size=32)
        batch = seqs[idx]
        b, l = batch.shape[:2]
        z = host.encode(batch.reshape(b * l, *batch.shape[2:])).reshape(b, l, -1)
        pred = host.forward_pred(z[:, :-1])
        loss = F.mse_loss(pred, z[:, 1:].detach())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    host.eval()
    for p in host.parameters():
        p.requires_grad_(False)
    return host


@torch.no_grad()
def surprise_stream(host: MiniHost, frames_t: torch.Tensor) -> np.ndarray:
    """Per-frame host one-step prediction error (label-free surprise)."""
    z = host.encode(frames_t)  # (L, dim)
    pred = host.forward_pred(z[:-1])
    err = ((pred - z[1:]) ** 2).mean(dim=-1).cpu().numpy()  # length L-1, for t=1..L-1
    out = np.zeros(frames_t.shape[0], dtype=np.float32)
    out[1:] = err
    return out


def saliency_stream(frames: np.ndarray) -> np.ndarray:
    """Hand-authored colour-saliency miner: per-frame max colour saturation."""
    f = frames.astype(np.float32)
    sat = f.max(axis=-1) - f.min(axis=-1)  # (L,H,W) chroma
    # miner scores a frame by its most colourful region (peak chroma)
    return sat.reshape(frames.shape[0], -1).max(axis=1)


@torch.no_grad()
def features_of_frame(host: MiniHost, frames_t: torch.Tensor, t: int) -> np.ndarray:
    return host.encode(frames_t[t:t + 1]).cpu().numpy()[0]


def run_mode(host, semantic, device, mode: str, n_episodes: int, seed: int) -> dict:
    rng = np.random.default_rng(777 + seed)
    labels = rng.integers(0, CLASSES, size=n_episodes)
    feats = {"surprise": [], "saliency": [], "random": [], "true_cue": []}
    write_choice = {"surprise": [], "saliency": [], "random": []}
    for ep in range(n_episodes):
        frames = render_episode(ep, int(labels[ep]), mode)
        ft = to_tensor(frames, device)
        s_sur = surprise_stream(host, ft)
        s_sal = saliency_stream(frames)
        # WRITE budget = 1: pick argmax over frames >=1
        t_sur = int(1 + np.argmax(s_sur[1:]))
        t_sal = int(1 + np.argmax(s_sal[1:]))
        t_random = int(rng.integers(1, LENGTH))
        write_choice["surprise"].append(t_sur)
        write_choice["saliency"].append(t_sal)
        write_choice["random"].append(t_random)
        patches = semantic.patches(ft)
        previous = patches[max(0, t_sur - 1)]
        patch_change = (patches[t_sur] - previous).square().mean(dim=-1)
        semantic_patch = int(torch.argmax(patch_change))
        random_patch = int(rng.integers(0, patches.shape[1]))
        feats["surprise"].append(
            patches[t_sur, semantic_patch].cpu().numpy())
        feats["saliency"].append(patches[t_sal].mean(0).cpu().numpy())
        feats["random"].append(
            patches[t_random, random_patch].cpu().numpy())
        feats["true_cue"].append(
            patches[CUE_FRAMES[-1]].mean(0).cpu().numpy())
    labels = np.asarray(labels)
    out = {"mode": mode, "n_episodes": int(n_episodes)}
    split = int(0.6 * n_episodes)
    for policy in ("surprise", "saliency", "random", "true_cue"):
        X = np.asarray(feats[policy])
        clf = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0))
        clf.fit(X[:split], labels[:split])
        pred = clf.predict(X[split:])
        out[f"{policy}_label_bacc"] = float(
            balanced_accuracy_score(labels[split:], pred))
    # how often each policy writes the actual cue frame (1..3)
    for policy in ("surprise", "saliency", "random"):
        wc = np.asarray(write_choice[policy])
        out[f"{policy}_writes_cue_frac"] = float(
            np.mean(np.isin(wc, CUE_FRAMES)))
        out[f"{policy}_writes_distractor_frac"] = float(
            np.mean(wc == DISTRACTOR_FRAME))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--dinov2", default=str(DEFAULT_DINOV2))
    ap.add_argument("--torch-home", default=str(DEFAULT_TORCH_HOME))
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    host = train_frozen_host(device)
    semantic = FrozenSemanticEncoder(
        device, Path(args.dinov2), Path(args.torch_home))
    results = {
        "schema": "cem_shortcut_probe_v1",
        "claim": ("label-free surprise WRITE retains identity on the colour "
                  "shortcut; the colour-saliency miner is captured by a "
                  "predictable salient distractor"),
        "chance_bacc": 1.0 / CLASSES,
        "modes": {},
    }
    for mode in ("shape_random_color", "color"):
        r = run_mode(host, semantic, device, mode, args.episodes, args.seed)
        results["modes"][mode] = r
        print(json.dumps(r, indent=2), flush=True)
    sr = results["modes"]["shape_random_color"]
    results["verdict"] = {
        "surprise_beats_saliency_on_shortcut": bool(
            sr["surprise_label_bacc"] > sr["saliency_label_bacc"] + 0.05),
        "surprise_retains_identity": bool(sr["surprise_label_bacc"] >= 0.6),
        "surprise_beats_random_semantic": bool(
            sr["surprise_label_bacc"] > sr["random_label_bacc"] + 0.05),
        "saliency_captured_by_distractor": bool(
            sr["saliency_writes_distractor_frac"] > 0.5),
    }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results["verdict"], indent=2), flush=True)
    print(f"wrote {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
