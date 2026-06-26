# Experiment Program: from "weak reject" to "weak accept"

*Response to the ICLR-style review (current estimate ~4.5–5/10). This plan operationalizes every required experiment into concrete runs in this codebase, prioritized by reviewer-impact ÷ effort. Target: borderline/weak accept (~6–7/10).*

## Reframing the claims (do first, free)
The review is right that two claims overreach. We adopt the weaker, defensible versions throughout `ICLR.md`:
- **From** "explicit memory, *not* hierarchy, is the missing primitive" → **to** "for short-context JEPA world models, explicit *timescale-controlled* memory is a simple primitive that *complements* hierarchy and enables controlled diagnosis of when information is available vs used."
- **From** "the memory horizon must match the gap" → **to** "the horizon controls cue *availability*; predictive *use* additionally depends on signal amplitude, interference, and whether the predictor learns to read the bank." (Directly supported by the τ-sweep, where τ=6 already gives availability 0.99 but usage 0.54.)
- Replace "two scalars" with "two scalar horizons + small zero-init linear read-outs (~2D² params)", and add a parameter-matched control.

## Priority table

| # | Experiment | Addresses review § | Effort | Retrain? | Status |
|---|---|---|---|---|---|
| **E1** | **Counterfactual memory swap (causal)** | #5, "one experiment above all" | low | no | implemented this turn |
| **E2** | Baseline suite (long-ctx / GRU / RSSM / SSM / FIFO-attn) | #3 | high | yes | designed |
| **E3** | Single- vs two- vs log-spaced K-bank EMA | #3, #7 | med | yes | designed (module ext.) |
| **E4** | Horizon law: 2D (Δ×τ) availability/usage/influence vs e^{-Δ/τ} | #4 | med | yes | designed (extends sweeps) |
| **E5** | ≥5 seeds + statistics + EMA-reset audit | #6 | low | yes | designed |
| **E6** | Standard benchmarks (POPGym Arcade ≥5 tasks, MiniGrid-Memory) | #2 | med-high | yes | POPGym partial done |
| **E7** | Downstream planning/control (CEM) + test-time bank ablation | #5 | high | no (uses trained) | designed |
| **E8** | Freeze-backbone / scale (frozen LeWM, V-JEPA2/DINO-WM feats) | #7 | high | yes | designed |

Recommended order: **E1 → E5 → E4 → E3 → E2 → E7 → E6 → E8**.

---

## E1 — Counterfactual memory swap (the headline causal experiment)
**Hypothesis.** The EMA banks *causally* control the prediction, not merely carry decodable info.
**Method** (`scripts/causal_swap.py`, no retraining — uses trained `both` models). For each episode *i*, predict the reveal-latent using *i*'s current frames+actions but **another episode j's memory banks** (cue[j]≠cue[i]). Train a cue probe on *real* reveal latents; apply to the predicted latent.
- `follow_self` = P(decode(pred with own memory) = cue[i]) — control.
- `follow_memory` = P(decode(pred with swapped memory) = **cue[j]**) — causal effect.
- `follow_current` = P(decode(swapped pred) = cue[i]) — should collapse.
**Expected / success.** `follow_memory ≫ chance` and `≫ follow_current`: the prediction tracks the *injected memory*, not the current frame → causal control. Counterfactual 2×2 (current A/B × memory A/B) reported.

## E2 — Strong, matched baselines (highest empirical priority)
Replace/augment the `none/short/long/both` axis with alternative memory mechanisms in the predictor, **compute- and parameter-matched**:
- **Long-context predictor**: `none` with window `h ∈ {3,9,21,39}` (already supported via `--history-len`; pos-embed scales). Tests "does EMA beat just enlarging h?"
- **GRU / LSTM** recurrent latent state feeding the predictor (new `RecurrentMemory` module, drop-in where `TwoTimescaleMemory` sits).
- **RSSM-lite**: GRU deterministic state (DreamerV3-style, no stochastic head) — standard MBRL memory.
- **Diagonal-SSM / RetNet-lite**: learned per-channel decay + input gate (our EMA is the fixed-α scalar case; this is the learned-matrix generalization).
- **FIFO latent cache + attention**: predictor attends over the last K stored latents (tests exponential compression vs raw storage).
Each as a new `--memory-impl {ema,gru,lstm,rssm,ssm,fifo}` flag; param-match by sizing hidden dims. **Result framing**: either EMA *matches* these while being simpler+interpretable, or it loses but uniquely gives controllable-horizon diagnostics — both publishable.

## E3 — Is *two* timescales right? Single vs log-spaced K-bank
Generalize `TwoTimescaleMemory` → `MultiTimescaleMemory(taus=[...])` (K leaky integrators, K zero-init read-outs).
- **Single-timescale sweep**: K=1, τ ∈ {2,4,8,16,32,64} → is one well-chosen τ enough?
- **Log-spaced bank**: τ = {2,4,8,16,32,64} → does a *fixed* multi-scale bank beat hand-picked (3,25) and remove the "must choose τ" brittleness (review #7)?
**Expected.** Log-spaced K-bank ≥ best single and ≥ two-bank, with no per-task tuning → fixes the learnable-α-doesn't-self-tune weakness by *spanning* horizons instead of learning one.

## E4 — Quantify the horizon law (turn a weakness into a contribution)
Full 2D grid over gap Δ ∈ {3,6,…,39} × τ ∈ {2,4,8,16,32,64} on T-Maze; for each cell measure **availability, usage, causal-influence**. Plot vs the exponential retention kernel `(1−α)^Δ = e^{−Δ/τ}`.
- Confirm availability ≈ exponential kernel.
- Fit the **usage threshold**: usage turns on when retained SNR `e^{−Δ/τ}` exceeds a critical value τ*-curve → an explicit "availability is necessary but not sufficient" law (the review's nuance, made quantitative).

## E5 — Statistical rigor (cheap, do early)
- **≥5 seeds** (10 for custom tasks) on the headline matrices; report **mean ± 95% CI** (bootstrap for probe accuracy), **paired t-test / Wilcoxon** across designs, per-seed scatter.
- Probe protocol appendix: train/test split, regularization, standardization, n.
- **EMA-reset audit**: verify the memory state resets at episode boundaries (no cross-episode leakage); add an explicit test (`test_memory.py`) and report.

## E6 — Standard memory benchmarks
- **POPGym Arcade** (done: CountRecall, AutoEncode) → expand to ≥5 tasks (add BattleShip, MineSweeper, Navigator), ≥5 seeds, with the matched baselines from E2.
- **MiniGrid-Memory** (gymnasium, light) and/or **Memory Gym**; **Memory Maze** if 3D budget allows.
- Target grid: ≥5 tasks × ≥5 seeds, CIs + significance.

## E7 — Downstream planning / control (closes the world-model loop)
On the goal-cued PO envs (tworoom_po, pusht_po, reacher_po): use each trained design inside **CEM/MPC** to reach the *remembered* goal; report **success rate, steps, regret**.
- Predicted ordering on long-gap tasks: `both ≈ long > short > none`; on short-gap: `short ≈ both > none`.
- **Test-time bank ablation** during planning (zero `m_fast`/`m_slow`) → causal behavioral break.
Requires a goal-conditioned cost that uses the *remembered* goal (the reveal latent / planned reach), and memory-aware rollout (already in `MemoryLeWorldModel.rollout_latents`).

## E8 — Freeze-backbone & scale
Train only memory+predictor on a **frozen** encoder: (a) frozen LeWM encoder, (b) frozen V-JEPA 2 / DINO-WM features. Shows the primitive is a *general* add-on to JEPA backbones, not a LeWM-specific tweak.

---

## Minimum bar for resubmission (review's explicit list)
1. Strong RNN/SSM/long-context baselines (**E2**), param/compute-matched.
2. ≥1 standard benchmark, ≥5 tasks × ≥5 seeds (**E6**).
3. Downstream planning/control (**E7**).
4. Causal memory ablation / swap (**E1**, **E7** ablation).
5. ≥5 seeds + stats (**E5**).
Single highest-value experiment (review's pick): **E1 + E7 on a standard PO benchmark** — counterfactual swap + closed-loop planning.
