# Validation Results — Two-Timescale Memory for JEPA World Models

> **Historical result only.** This page reports the original single-seed,
> 30-epoch fixed-EMA study and is not the current ICLR evidence. The completed
> frozen LeWM+V8 V18 study covers 200/200 cells and returns
> **`CONFIRMATION_FAILED`**; see the [analysis-rendered manuscript](ICLR.md),
> [write-once decision](../paper/review_artifact/confirmation_analysis.json),
> and [full independent audit](V18_FINAL_AUDIT.json).

*Single-seed validation run (seed 0), 30 epochs/cell, fixed horizons τ_fast=3, τ_slow=25, 4 GPUs.*
*wandb: project `lewm-memory`. Figures in `outputs/mem/`. Regenerate: `scripts/run_all.sh` → `scripts/analyze_results.py`.*

## Headline
On four partially-observable memory-stressing environments, adding the two-timescale EMA memory **reduces world-model prediction error by up to ~80%, drives the decision via the correct timescale, and has ~zero effect on the fully-observable control** — exactly the dissociation the method predicts. Result strength: **strong & clean** on `tmaze`/`occlusion`/`tworoom`; **positive but noisier** on `recall`/`distractor` (stochastic envs; needs multi-seed).

## 1. Availability — the math, confirmed (cleanest result)
Linear probe decoding the cue from each stream over time (design = `both`):

- **`tmaze` (long gap Δ≈21):** the memoryless encoder `z` drops to chance the instant the cue leaves frame (t=3); the **fast** bank (τ=3) holds it for a while then **decays to chance by the decision** (its red curve traces the exponential kernel); the **slow** bank (τ=25) stays at **1.0 across the entire 21-step gap**. → *only long-term memory survives a long gap.* (`outputs/mem/lewm-tmaze-both-s0/probe_cue_over_time_e30.png`)
- **`occlusion` (short gap Δ≈5):** `z` drops to chance only *during* the occlusion; the **fast** bank stays high and **bridges the short gap** (where in tmaze it had decayed). → *short-term memory suffices for short gaps.* (`outputs/mem/lewm-occlusion-both-s0/...`)

Decode accuracy of the cue at the decision step (delay), design=`both`:

| env | z (memoryless) | m_fast (τ=3) | m_slow (τ=25) |
|---|---|---|---|
| tmaze (long) | 0.44 (chance) | 0.44 (chance) | **1.00** |
| distractor (long+interf.) | 0.48 | 0.49 (chance) | **1.00** |
| occlusion (short) | 0.48 (chance) | **0.92** | 0.98 |
| recall (mixed, 3-class) | 0.35 (chance .33) | 0.33 | **0.56** |

## 2. Usage — does the *decision* use memory? (corrected, matched probe)
Train a probe on the model's *predicted* reveal-latent, test on held-out predicted latents (the earlier cross-distribution probe was a metric artifact — see note). Accuracy = is the cue decodable from the model's prediction? (`summary_decision_corrected.png`)

| env | none | short | long | both | chance |
|---|---|---|---|---|---|
| tmaze | 0.57 | 0.58 | **0.84** | **0.88** | 0.50 |
| distractor | 0.50 | 0.56 | **0.94** | **0.86** | 0.50 |
| occlusion | 0.52 | 0.49 | **0.68** | **0.63** | 0.50 |
| recall | 0.35 | 0.33 | 0.34 | **0.43** | 0.33 |

→ **long/both make the cue decodable from the model's prediction; none/short stay near chance** on the long-horizon tasks. Memory measurably shapes the decision.

## 3. Prediction error — memory helps, control unaffected
Validation next-latent MSE (same held-out val set within each env, from training logs):

| env | none | short | long | both | kind |
|---|---|---|---|---|---|
| **tmaze** | 1.114 | 0.519 | 0.471 | **0.414** | long → monotone: more memory, lower error |
| **occlusion** | 0.696 | **0.430** | 0.448 | 0.443 | short → a *little* (fast) memory captures most gain |
| **recall** | 1.211 | 0.348 | 0.423 | **0.146** | mixed → memory helps hugely |
| **distractor** | 0.539 | 0.385 | 0.590 | **0.026** | long+interf. → both ≫ long-alone (needs fast for recency) |
| **tworoom** | 0.478 | — | — | 0.478 | **Markovian control: identical → memory gives no advantage** |

The `tworoom` control is the critical sanity check: with the same val set, `none` and `both` are **indistinguishable (0.478 vs 0.478)**, so the gains on the other envs are real, not a generic capacity effect.

## 4. Honest caveats
- **Single seed.** tmaze/occlusion/tworoom are clean; `recall` and `distractor` aggregate-MSE are noisier (`both` is not always the best cell, and these envs have stochastic elements that add irreducible error). Multi-seed (≥3) is the obvious next step.
- **Aggregate per-frame MSE is roughly uniform over time** (memory shifts the whole curve down rather than spiking at the reveal), because the cue occupies a small sub-space of the global latent — the cue-specific effect is best read from the *probes* (§1–2), not the aggregate MSE. (`analysis_mse_by_time.png`)
- **Metric fix:** the original `cue_acc_from_prediction` trained the probe on encoder latents but tested on predictor outputs (a distribution shift) and read ~chance for every design; the matched probe in §2 is the corrected version.
