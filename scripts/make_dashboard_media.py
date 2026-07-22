#!/usr/bin/env python3
"""Build rendered-rollout and evidence-age cue animations for the research dashboard.

This is a *standalone* media generator for ``docs/mesm_nvidia_plan.html``.  It does
not train anything and needs no GPU: it assembles frames that are already cached in
the OGBench render caches into animated GIFs (and MP4s if ``ffmpeg`` is available).

Two artefacts are produced:

1. Rollout animations -- a representative episode from a render cache turned into a
   looping, upscaled GIF/MP4 so a reader can see a *real* environment moving.

2. An annotated "evidence-age" cue animation -- the same clean episode with the
   4-way cue card injected at frames 1..LAST_CUE_FRAME (via the paper's own
   ``inject_cue_sequence``) and then annotated frame-by-frame with:
     * a "CUE VISIBLE" banner during the injection frames,
     * a sliding "legal context window (K)" box on a timeline strip, and
     * a growing "evidence age" counter (frames since the cue was last shown)
       up to a marked readout frame.

The cue-drawing utilities are reused (not re-implemented) from the paper runner
``scripts/run_masked_evidence_jepa_ogbench.py`` so this stays faithful to the
experiment.  We only *read* that file; we never edit it.

Usage
-----
    .venv/bin/python scripts/make_dashboard_media.py            # build everything
    .venv/bin/python scripts/make_dashboard_media.py --no-mp4   # GIF only

All outputs land in ``docs/assets/``.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# CPU-only: assembling cached frames needs no GPU, and we want to avoid touching a
# busy GPU merely by importing torch inside the paper runner.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
PAPER_RUNNER = ROOT / "scripts" / "run_masked_evidence_jepa_ogbench.py"

# 140-frame caches (preferred: longer rollouts) and the 22-frame fallback caches.
LONG_CACHE = ROOT / "outputs" / "paper_c_agescale_v1" / "cache"
SHORT_CACHE = ROOT / "outputs" / "multiview_patchset_color_jepa_native_v1" / "cache"


# --------------------------------------------------------------------------- #
# Reuse the paper's cue utilities without executing its training code.
# --------------------------------------------------------------------------- #
def _load_paper_utils():
    """Import cue helpers from the paper runner (read-only reuse)."""
    spec = importlib.util.spec_from_file_location("paper_runner", PAPER_RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


# --------------------------------------------------------------------------- #
# Fonts
# --------------------------------------------------------------------------- #
def _load_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    try:
        import matplotlib

        candidates.append(str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"))
    except Exception:
        pass
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


# --------------------------------------------------------------------------- #
# Cache helpers
# --------------------------------------------------------------------------- #
def cache_for(env: str, prefer_long: bool = True) -> Path | None:
    """Return the render-cache path for ``env``, preferring the 140-frame cache."""
    order = [LONG_CACHE, SHORT_CACHE] if prefer_long else [SHORT_CACHE, LONG_CACHE]
    for base in order:
        candidate = base / env / "render_cache.npz"
        if candidate.exists():
            return candidate
    return None


def load_cache(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def pick_episode(frames: np.ndarray, prefer: int | None = None) -> int:
    """Pick a representative episode: the one with the most on-screen motion.

    Motion is the mean absolute inter-frame difference across the episode, which
    surfaces trajectories where the agent/object actually moves rather than idles.
    """
    if prefer is not None and 0 <= prefer < frames.shape[0]:
        return int(prefer)
    # Subsample episodes for speed if there are many.
    n = frames.shape[0]
    idx = np.arange(n)
    diffs = np.abs(np.diff(frames[idx].astype(np.int16), axis=1)).mean(axis=(1, 2, 3, 4))
    return int(idx[int(np.argmax(diffs))])


def upscale(frame: np.ndarray, size: int, smooth: bool = False) -> Image.Image:
    resample = Image.Resampling.BILINEAR if smooth else Image.Resampling.NEAREST
    return Image.fromarray(frame.astype(np.uint8)).resize((size, size), resample)


# --------------------------------------------------------------------------- #
# Writers (GIF + optional MP4)
# --------------------------------------------------------------------------- #
def save_gif(images: list[Image.Image], out: Path, fps: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(round(1000.0 / fps))
    images[0].save(
        out,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )


def save_mp4(images: list[Image.Image], out: Path, fps: int) -> bool:
    """Encode an MP4 with ffmpeg if available. Returns True on success."""
    if shutil.which("ffmpeg") is None:
        return False
    # Pad frames to even dimensions (yuv420p / H.264 requirement).
    w, h = images[0].size
    w2, h2 = w + (w % 2), h + (h % 2)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-an",
        "-vf", f"pad={w2}:{h2}:0:0:color=black",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for img in images:
        proc.stdin.write(np.asarray(img.convert("RGB"), dtype=np.uint8).tobytes())
    proc.stdin.close()
    return proc.wait() == 0


# --------------------------------------------------------------------------- #
# Task 1: plain rollout animation of a real environment
# --------------------------------------------------------------------------- #
def make_rollout(
    env: str,
    out_stem: str,
    *,
    size: int = 256,
    fps: int = 10,
    max_frames: int = 140,
    episode: int | None = None,
    want_mp4: bool = True,
) -> dict:
    cache_path = cache_for(env, prefer_long=True)
    if cache_path is None:
        return {"env": env, "ok": False, "reason": "no render cache found"}
    cache = load_cache(cache_path)
    frames = cache["frames"]
    ep = pick_episode(frames, prefer=episode)
    seq = frames[ep][:max_frames]
    src_h = int(seq.shape[1])

    font_tag = _load_font(13)
    images: list[Image.Image] = []
    total = seq.shape[0]
    for t, raw in enumerate(seq):
        img = upscale(raw, size, smooth=False).convert("RGB")
        draw = ImageDraw.Draw(img)
        # Small caption strip: env + frame counter.
        draw.rectangle([0, size - 22, size, size], fill=(0, 0, 0))
        draw.text((6, size - 19), env, font=font_tag, fill=(251, 212, 91))
        counter = f"t={t:>3}/{total - 1}"
        w = draw.textlength(counter, font=font_tag)
        draw.text((size - w - 6, size - 19), counter, font=font_tag, fill=(251, 251, 249))
        images.append(img)

    gif_path = ASSETS / f"{out_stem}.gif"
    save_gif(images, gif_path, fps=fps)
    result = {
        "env": env,
        "ok": True,
        "cache": str(cache_path.relative_to(ROOT)),
        "episode": ep,
        "frames": total,
        "src_size": src_h,
        "gif": str(gif_path.relative_to(ROOT)),
    }
    if want_mp4:
        mp4_path = ASSETS / f"{out_stem}.mp4"
        if save_mp4(images, mp4_path, fps=fps):
            result["mp4"] = str(mp4_path.relative_to(ROOT))
    return result


# --------------------------------------------------------------------------- #
# Task 2: annotated evidence-age cue animation
# --------------------------------------------------------------------------- #
CUE_NAMES = ["RED", "GREEN", "BLUE", "YELLOW"]


def _timeline_strip(
    draw: ImageDraw.ImageDraw,
    *,
    x0: int,
    y0: int,
    width: int,
    height: int,
    total: int,
    t: int,
    last_cue: int,
    k_window: int,
    readout: int,
    cue_color: tuple[int, int, int],
) -> None:
    """Draw a per-frame timeline: cue frames, sliding legal window, readout, cursor."""
    n = total
    cell = width / n
    top = y0
    bot = y0 + height
    # Base track.
    draw.rectangle([x0, top, x0 + width, bot], fill=(30, 36, 48), outline=(120, 130, 145))
    # Cue frames (1..last_cue) painted in the cue colour.
    for f in range(1, last_cue + 1):
        cx0 = x0 + f * cell
        draw.rectangle([cx0, top, cx0 + cell, bot], fill=cue_color)
    # Readout frame marker.
    rx = x0 + readout * cell
    draw.rectangle([rx, top, rx + cell, bot], fill=(251, 212, 91))
    # Sliding legal context window [t-k+1 .. t].
    ws = max(0, t - k_window + 1)
    we = t + 1
    draw.rectangle(
        [x0 + ws * cell, top - 4, x0 + we * cell, bot + 4],
        outline=(255, 255, 255),
        width=2,
    )
    # Current-frame cursor.
    cur = x0 + t * cell + cell / 2
    draw.line([cur, top - 8, cur, bot + 8], fill=(255, 90, 90), width=2)


def make_cue_animation(
    envs: list[tuple[str, str]],
    out: Path,
    *,
    size: int = 220,
    fps: int = 6,
    k_window: int = 3,
    age: int = 15,
    tail: int = 4,
    want_mp4: bool = True,
) -> dict:
    """Build one annotated GIF that teaches the evidence-age retention demand.

    For each (env, kind) we take a clean episode, inject the 4-way cue at frames
    1..LAST_CUE_FRAME, then annotate the walk-forward up to the readout frame
    (endpoint = LAST_CUE_FRAME + age) plus a short tail.
    """
    utils = _load_paper_utils()
    last_cue = int(utils.LAST_CUE_FRAME)
    readout = last_cue + age
    end = readout + tail
    palette = utils.PALETTE

    canvas_w = size + 80
    canvas_h = size + 168
    rx = (canvas_w - size) // 2
    font_title = _load_font(14)
    font_body = _load_font(13)
    font_small = _load_font(11)
    font_big = _load_font(32)

    images: list[Image.Image] = []
    used = []
    for env, kind in envs:
        cache_path = cache_for(env, prefer_long=True)
        if cache_path is None:
            continue
        cache = load_cache(cache_path)
        frames = cache["frames"]
        if frames.shape[1] <= readout:
            # Not long enough for this age; skip gracefully.
            continue
        ep = pick_episode(frames)
        label = int(cache["cue_labels"][ep]) % len(palette)
        position = int(cache["cue_positions"][ep]) % 4
        injected = utils.inject_cue_sequence(frames[ep].copy(), label, position)
        cue_color = tuple(int(v) for v in palette[label])
        used.append({"env": env, "kind": kind, "episode": ep, "cue": CUE_NAMES[label]})

        for t in range(0, min(end + 1, injected.shape[0])):
            canvas = Image.new("RGB", (canvas_w, canvas_h), (17, 20, 27))
            draw = ImageDraw.Draw(canvas)
            # Env render.
            render = upscale(injected[t], size, smooth=False).convert("RGB")
            ry = 44
            canvas.paste(render, (rx, ry))
            draw.rectangle([rx, ry, rx + size, ry + size], outline=(80, 90, 105))

            # Header.
            draw.text((12, 8), f"{env}", font=font_title, fill=(251, 251, 249))
            draw.text((12, 26), f"{kind}  ·  cue={CUE_NAMES[label]}", font=font_small, fill=(156, 163, 175))

            # Status banner over the render.
            cue_visible = 1 <= t <= last_cue
            if cue_visible:
                banner, bcol = "CUE VISIBLE", (230, 57, 70)
            elif t < readout:
                banner, bcol = "CUE GONE — HOLD IN MEMORY", (52, 152, 219)
            else:
                banner, bcol = "READOUT", (251, 212, 91)
            bh = 22
            draw.rectangle([rx, ry, rx + size, ry + bh], fill=bcol)
            btxt = banner
            bw = draw.textlength(btxt, font=font_body)
            tcol = (17, 20, 27) if bcol == (251, 212, 91) else (255, 255, 255)
            draw.text((rx + (size - bw) / 2, ry + 4), btxt, font=font_body, fill=tcol)

            # Evidence-age counter (frames since the cue was last shown).
            evidence_age = max(0, t - last_cue)
            in_window = t - k_window + 1 <= last_cue  # cue still inside legal window
            age_y = ry + size + 12
            age_col = (46, 204, 113) if in_window else (251, 212, 91)
            draw.text((12, age_y), "EVIDENCE AGE", font=font_small, fill=(156, 163, 175))
            draw.text((12, age_y + 13), f"{evidence_age}", font=font_big, fill=age_col)
            num_w = draw.textlength(f"{evidence_age}", font=font_big)
            state = "in legal window" if in_window else "outside window\nmust recall from memory"
            draw.multiline_text((24 + num_w, age_y + 16), state, font=font_small, fill=age_col, spacing=3)
            draw.text(
                (12, age_y + 52),
                f"legal context K={k_window}  ·  readout at age={age}",
                font=font_small,
                fill=(156, 163, 175),
            )

            # Timeline strip.
            strip_y = canvas_h - 34
            _timeline_strip(
                draw,
                x0=12,
                y0=strip_y,
                width=canvas_w - 24,
                height=12,
                total=min(end + 1, injected.shape[0]),
                t=t,
                last_cue=last_cue,
                k_window=k_window,
                readout=readout,
                cue_color=cue_color,
            )
            draw.text((12, strip_y + 16), "t=0", font=font_small, fill=(120, 130, 145))
            rd = "readout"
            rw = draw.textlength(rd, font=font_small)
            draw.text((canvas_w - 12 - rw, strip_y + 16), rd, font=font_small, fill=(251, 212, 91))

            images.append(canvas)

    if not images:
        return {"ok": False, "reason": "no suitable long cache for requested age"}

    save_gif(images, out, fps=fps)
    result = {
        "ok": True,
        "gif": str(out.relative_to(ROOT)),
        "segments": used,
        "frames": len(images),
        "k_window": k_window,
        "age": age,
        "readout_frame": readout,
    }
    if want_mp4:
        mp4_path = out.with_suffix(".mp4")
        if save_mp4(images, mp4_path, fps=fps):
            result["mp4"] = str(mp4_path.relative_to(ROOT))
    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-mp4", action="store_true", help="Skip MP4 encoding even if ffmpeg exists.")
    p.add_argument("--rollout-fps", type=int, default=10)
    p.add_argument("--cue-fps", type=int, default=6)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    ASSETS.mkdir(parents=True, exist_ok=True)
    want_mp4 = not args.no_mp4

    results = []
    # Task 1: rollouts (navigation + manipulation).
    results.append(
        make_rollout("pointmaze-large-navigate-v0", "rollout_pointmaze_large",
                     fps=args.rollout_fps, want_mp4=want_mp4)
    )
    results.append(
        make_rollout("cube-single-play-v0", "rollout_cube_single",
                     fps=args.rollout_fps, want_mp4=want_mp4)
    )

    # Task 2: annotated evidence-age cue animation (one nav + one manip env).
    cue_res = make_cue_animation(
        [("pointmaze-large-navigate-v0", "navigation"), ("cube-single-play-v0", "manipulation")],
        ASSETS / "cue_animation.gif",
        fps=args.cue_fps,
        want_mp4=want_mp4,
    )
    results.append(cue_res)

    print("\n=== dashboard media generation report ===")
    for r in results:
        print(r)
    # Fail loudly if nothing was produced.
    if not any(r.get("ok") for r in results):
        print("ERROR: no media produced", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
