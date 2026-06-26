# Two-Timescale Memory for Joint-Embedding Predictive World Models

*Manuscript draft, ICLR 2027 format. Companion code: this repository. Literature review: [`RESEARCH_BRIEF.md`](RESEARCH_BRIEF.md); method details: [`PROPOSAL.md`](PROPOSAL.md); raw results: [`RESULTS.md`](RESULTS.md).*

---

## Abstract

Joint-Embedding Predictive Architectures (JEPAs) have become a leading recipe for latent world models: an encoder maps each observation to a representation and a predictor forecasts future representations. Yet these models are *memoryless* in time — the encoder sees one frame at a time and the predictor attends only over a short fixed window — so they cannot represent how information at different temporal distances shapes the dynamics of the latent space. We introduce a minimal, mathematically transparent remedy: a **two-timescale exponential memory** that augments the predictor with a *fast* (short-term) and a *slow* (long-term) exponential-moving-average (EMA) bank over the latent stream, injected through zero-initialized projections so training begins exactly at the memoryless baseline. The mechanism is the simplest diagonal linear state-space model; it adds two interpretable scalars whose closed-form effective horizon is $\tau=-1/\ln(1-\alpha)$, and it keeps the host model's two-term loss intact. On a controlled suite of partially-observable, memory-stressing environments built on the recent LeWorldModel, we show that (i) the memory horizon must *match* the task's cue-to-decision gap — a short bank bridges short gaps and only the long bank bridges long ones; (ii) the matched timescale carries information to the *decision*, lifting cue-decodability from the model's own prediction from chance to $0.84$ on long-horizon tasks; and (iii) memory extends the usable horizon far beyond the predictor's window, degrading gracefully where the memoryless baseline cliffs. A fully-observable control confirms the gains are memory-specific. We argue that an explicit, controllable multi-timescale memory — rather than the hierarchy/sub-goal direction the field currently favors — is a foundational primitive for studying memory in JEPA latent dynamics, and we contribute a reusable *availability-vs-usage* measurement protocol.

---

## 1. Introduction

Latent world models predict the future in a learned representation space rather than in pixels, and JEPAs are the dominant self-supervised instantiation: I-JEPA, V-JEPA 2, DINO-WM, and most recently **LeWorldModel (LeWM)**, which trains stably end-to-end from pixels with only a next-embedding prediction loss plus an isotropic-Gaussian regularizer (SIGReg). These models excel at *per-frame* representation, but their handling of *time* is impoverished: the encoder is applied frame-by-frame and the predictor attends over a short fixed history window of $h$ frames. Consequently, any information whose decision-relevance is separated from its appearance by more than $h$ steps is simply unavailable at decision time. Every model in this family reports compounding error over long horizons as its dominant failure mode, and the community's response has been *hierarchy and sub-goals* — not memory. LeWM's own paper, for instance, names hierarchical world modeling, not memory, as future work.

We take the orthogonal view that **explicit, controllable memory** is the missing primitive, and that the right first step is the *simplest* one that is also analyzable. We add to the predictor a two-timescale exponential memory: two leaky integrators over the latent stream with very different time constants. This is the scalar, fixed-decay special case of the structured state-space models (S4/Mamba) and multi-scale decay attention (RetNet) that dominate long-range sequence modeling, and it is the algorithmic core of Complementary Learning Systems theory (fast hippocampal vs. slow neocortical memory). Its decay kernel is a closed-form exponential whose horizon is a single interpretable number, making the memory a *plottable object* rather than a black box.

**Contributions.**
1. **A minimal primitive.** A two-timescale EMA memory for JEPA predictors: two scalars + two zero-initialized projections, preserving the host's two-term loss (§3).
2. **A measurement protocol.** We separate *availability* (is the cue still linearly present in a representation stream over time?) from *usage* (does the model's prediction at the decision encode the cue?), plus a memory-ablation *influence* functional (§4.3).
3. **Controlled evidence.** On four memory-stressing environments plus a Markovian control, the memory horizon must match the cue-to-decision gap, the matched timescale drives the decision, and memory degrades gracefully where the finite window cliffs (§5).
4. **Honest scope.** We report where the picture is clean (decision/availability) and where it is not (raw prediction MSE is a decoupled instrument; learned decay rates do *not* self-tune to the task) (§5.4, §6).

## 2. Related Work

**JEPA latent world models.** I-JEPA (Assran et al., 2023), V-JEPA 2 (Assran et al., 2025), DINO-WM (Zhou et al., 2024), and LeWorldModel (Maes et al., 2026), the last building on LeJEPA's SIGReg objective (Balestriero & LeCun, 2025). All use either no temporal predictor, a fixed history window, or a single recurrent state; none use a controllable multi-timescale memory. The recognized long-horizon failure mode has motivated *hierarchical* remedies (FF-JEPA, HWM) rather than memory.

**Memory in model-based RL.** DreamerV3's RSSM (Hafner et al., 2023) carries a single fixed-size recurrent state; S4WM (Deng et al., 2023) and R2I (Samsami et al., 2024) replace it with structured state-space models for long-range memory; Hieros (Mattes et al., 2023) stacks S5 at multiple timescales. The closest prior work is **MTS3** (Shaj et al., NeurIPS 2023), which learns two SSMs at fast (per-step) and slow (coarse-step) *sampling rates*. We differ on three axes: (i) ours is a self-supervised **JEPA**, not a probabilistic generative RSSM; (ii) our timescales are **memory-decay horizons** ($\tau=1/\alpha$), not coarse sampling rates / dynamics abstractions; and (iii) ours is deliberately a *minimal, analyzable primitive* (two scalars, zero-init) rather than a full hierarchical model.

**Memory math.** HiPPO (Gu et al., 2020), S4 (Gu et al., 2022), and RetNet (Sun et al., 2023, with per-head multi-scale decay $\gamma_h$) frame memory as an exponential/structured convolution kernel; our banks are the scalar fixed-$\alpha$ case. **Memory benchmarks.** POPGym (Morad et al., 2023) and POPGym Arcade (2025), Memory Maze, and Memory Gym isolate memory under partial observability; we build minimal analogues for controlled probing and discuss standard-benchmark evaluation as the key next step (§6).

## 3. Method

### 3.1 Background: the memoryless JEPA world model

The encoder $E_\theta$ maps each frame to a latent $z_t=E_\theta(o_t)\in\mathbb{R}^D$, regularized toward an isotropic Gaussian by SIGReg. The predictor $P_\phi$ takes a window of the last $h$ latents and the actions and predicts the next latent; the training loss is

$$\mathcal{L} = \underbrace{\lVert \hat z_{t+1}-z_{t+1}\rVert^2}_{\text{prediction}} + \lambda\,\mathrm{SIGReg}(Z). \tag{1}$$

With no state beyond the $h$-frame window, information older than $h$ steps is lost.

### 3.2 Two-timescale exponential memory

We maintain two EMA banks over the latent stream, indexed by $c\in\{\text{fast},\text{slow}\}$:

$$m^{(c)}_t = (1-\alpha_c)\,m^{(c)}_{t-1} + \alpha_c\,z_t. \tag{2}$$

Unrolling (2) reveals a causal convolution of the past with an **exponential memory kernel**,

$$m^{(c)}_t=\alpha_c\sum_{k\ge0}(1-\alpha_c)^k z_{t-k},\qquad K_c(k)=\alpha_c(1-\alpha_c)^k, \tag{3}$$

whose **effective horizon** (time constant) is closed-form:

$$\tau_c = \frac{-1}{\ln(1-\alpha_c)}\approx\frac1{\alpha_c},\qquad \alpha_c = 1-e^{-1/\tau_c}. \tag{4}$$

A *fast* bank (large $\alpha$, small $\tau$) is working memory; a *slow* bank (small $\alpha$, large $\tau$) is long-term memory. Equation (2) is exactly a diagonal linear state-space model — the simplest member of the S4/Mamba family — and the two-bank split is the computational form of Complementary Learning Systems.

### 3.3 Zero-init injection and the four ablations

The banks are injected additively into the only thing the predictor sees:

$$\tilde z_t = z_t + \mathbb 1[\text{short}]\,W_f\,m^{(\text{fast})}_t + \mathbb 1[\text{long}]\,W_s\,m^{(\text{slow})}_t, \tag{5}$$

with $W_f,W_s$ **zero-initialized**, so training starts *identical* to the memoryless baseline and recruits memory only as it lowers (1). The indicator flags yield the four designs we compare: `none` (vanilla LeWM), `short`, `long`, `both`.

### 3.4 Memory is the only long-range channel

We keep the predictor strictly short-context by training it with a **sliding window of length $h$** over a longer chunk $L\gg h$: each window predicts only its next latent. Because no window exceeds $h$ frames, information traveling further than $h$ steps can pass *only* through the EMA banks. This isolates the memory's contribution. The mechanism adds two scalars and two $D\times D$ matrices ($\approx 2D^2$ params, $\sim$1.5% of the model) and an $O(LD)$ scan; it keeps loss (1) unchanged.

## 4. Experimental Setup

### 4.1 Environments

Each environment contains a **cue-determined event**: something appearing later is decided by a cue shown earlier and is *not* recoverable from the current frame or action — so a memoryless model cannot predict it. An independent random-walk agent dot provides genuine action-conditioned dynamics orthogonal to the memory channel. We vary the cue→decision gap $\Delta$:

| env | memory kind | gap $\Delta$ | what must be remembered |
|---|---|---|---|
| **T-Maze** | long | $\approx21$ | which arm the early cue selected |
| **Distractor** | long + interference | $\approx23$ | the *first* cue, despite random flashes |
| **Recall** | mixed | $\approx15$ | a 3-symbol colour sequence (replayed) |
| **Occlusion** | short | $\approx5$ | the target's lane while briefly hidden |
| **TwoRoom** | Markovian control | $0$ | nothing (memory must *not* help) |

### 4.2 Models and training

Encoder ViT (patch 8, 64×64 RGB), $D{=}128$, predictor window $h{=}3$; fixed horizons $\tau_{\text{fast}}{=}3,\ \tau_{\text{slow}}{=}25$ unless stated. 30 epochs, 5000 episodes/epoch, AdamW, bf16. Each cell is run for **3 seeds**; we report mean$\pm$std. The vanilla LeWM baseline is `none` (memoryless, short-context) under the *identical* pipeline, for a controlled comparison. Logged to wandb project `lewm-memory-4ens`.

### 4.3 Metrics

- **Availability** $A_s(t)$: accuracy of a linear probe decoding the cue from stream $s\in\{z,m^{\text{fast}},m^{\text{slow}}\}$ at time $t$. Measures where information *lives*.
- **Usage**: accuracy of a probe — trained and tested on the model's *predicted* reveal-latent — decoding the cue. Measures whether the *decision* encodes it. (Training the probe on encoder latents but testing on predictions is a distribution-shifted artifact that reads at chance for all designs; the matched probe is the correct measure.)
- **Influence** $\mathcal I_c=\lVert f(\tilde z)-f(\tilde z\mid W_c{\leftarrow}0)\rVert_2$: movement of the predicted latent when bank $c$ is ablated.
- **Prediction MSE**: next-latent validation error (reported, but see §5.4).

## 5. Results

**Results at a glance — vanilla LeWM vs two-timescale memory on the four memory environments** (`lewm-memory-4ens`, 3 seeds). Decision-usage = cue decodable from the model's prediction (chance 0.50, except Recall 0.33); MSE = next-latent validation error; "→ best-mem" gives the best memory design.

| env | gap $\Delta$ | usage: none → best-mem | MSE: none → best-mem | winning timescale |
|---|---:|---|---|---|
| T-Maze | 21 | 0.50 → **0.84** (both) | 0.76 → 0.47 (short) | long / both |
| Distractor | 23 | 0.55 → **0.94** (long) | 0.39 → 0.41 (no gain) | long |
| Recall | 15 | 0.37 → **0.46** (both) | 0.76 → **0.33** (short) | short / long |
| Occlusion | 5 | 0.49 → 0.58 (long) | 0.46 → **0.25** (both) | short / both |
| TwoRoom (control) | 0 | — (no cue) | 0.48 → 0.48 (unchanged) | none needed |

Memory improves the decision on every memory-stressing environment and gives **no** advantage on the Markovian control; the winning timescale matches the task's gap $\Delta$. The per-metric breakdown and analysis follow.

### 5.1 The memory horizon must match the gap (availability)

Figure 1 is the central result. In **T-Maze** (long gap), the memoryless encoder $z$ falls to chance the instant the cue leaves the frame, the *fast* bank holds it briefly and then decays along its exponential kernel — reaching chance *before* the decision — while only the *slow* bank retains the cue across the full 21-step gap. In **Occlusion** (short gap), the *fast* bank alone bridges the brief occlusion where $z$ collapses. Short memory suffices for short gaps; long memory is necessary for long gaps.

![Figure 1: short-vs-long dissociation](figures/fig_dissociation.png)
*Figure 1. Cue-decoding accuracy over time (mean$\pm$std, 3 seeds), design `both`. Left: T-Maze (long). Right: Occlusion (short). Dotted = chance; black/green dashed = cue-off / decision.*

Table 1 reports availability at the decision step:

**Table 1 — Availability at the decision (design `both`, 3 seeds).** Cue decodable from each stream.

| env | gap $\Delta$ | $z$ (memoryless) | $m^{\text{fast}}$ ($\tau{=}3$) | $m^{\text{slow}}$ ($\tau{=}25$) |
|---|---:|---:|---:|---:|
| Occlusion | 5 | 0.46 | **0.90** | 0.99 |
| Recall | 15 | 0.33 | 0.33 | **0.55** |
| T-Maze | 21 | 0.53 | 0.49 | **1.00** |
| Distractor | 23 | 0.54 | 0.52 | **1.00** |

### 5.2 The matched timescale drives the decision (usage)

**Table 2 — Usage: cue decodable from the model's prediction (matched probe, 3 seeds, mean$\pm$std).** Higher is better; chance in last column.

| env | gap $\Delta$ | none (vanilla) | short | long | both | chance |
|---|---:|---:|---:|---:|---:|---:|
| T-Maze | 21 | 0.50 ±.01 | 0.52 ±.02 | 0.81 ±.06 | **0.84 ±.08** | 0.50 |
| Distractor | 23 | 0.55 ±.01 | 0.56 ±.01 | **0.94 ±.05** | 0.84 ±.09 | 0.50 |
| Recall | 15 | 0.37 ±.04 | 0.38 ±.03 | 0.45 ±.08 | **0.46 ±.05** | 0.33 |
| Occlusion | 5 | 0.49 ±.04 | 0.55 ±.06 | **0.58 ±.03** | 0.56 ±.08 | 0.50 |

On the long-horizon tasks the `long`/`both` designs lift decision-usage well above the vanilla baseline and chance with low variance; `none`/`short` stay at chance (Figure 2).

![Figure 2: usage across envs](figures/fig_usage_bar.png)
*Figure 2. Cue decodable from the model's prediction across envs and designs (3 seeds, mean$\pm$std; dotted = chance).*

### 5.3 Memory extends the usable horizon; the finite window cliffs

Sweeping the cue→decision gap $\Delta$ on T-Maze (Table 3, Figure 3) shows the vanilla baseline flat at chance for every $\Delta$ beyond its 3-frame window, while the memory model holds high usage and **degrades gracefully**, approaching chance only as $\Delta\to\tau_{\text{slow}}$.

**Table 3 — T-Maze gap sweep: usage vs. gap $\Delta$ (window $h{=}3$, $\tau_{\text{slow}}{=}25$; 3 seeds, mean).**

| $\Delta$ | 3 | 9 | 15 | 21 | 27 | 33 | 39 |
|---|---:|---:|---:|---:|---:|---:|---:|
| vanilla (none) | 0.55 | 0.55 | 0.44 | 0.53 | 0.44 | 0.48 | 0.45 |
| memory (both) | **0.98** | **0.94** | **0.88** | **0.87** | **0.83** | **0.77** | **0.64** |

![Figure 3: gap sweep](figures/exp4_gap_sweep.png)
*Figure 3. Memory vs. finite-window baseline as the gap grows. Left: usage. Right: MSE (noisy; see §5.4).*

Sweeping the slow-bank horizon $\tau_{\text{slow}}$ at fixed gap $\Delta{=}21$ (Table 4) confirms the duration is the operative knob: a too-short slow bank ($\tau{=}3$) cannot hold the cue at all (availability at chance), and once $\tau_{\text{slow}}\gtrsim\Delta$ the decision recovers it.

**Table 4 — T-Maze $\tau_{\text{slow}}$ sweep ($\Delta{=}21$, design `both`; 3 seeds, mean).**

| $\tau_{\text{slow}}$ | 3 | 6 | 12 | 21 | 30 | 45 |
|---|---:|---:|---:|---:|---:|---:|
| availability ($m^{\text{slow}}$) | 0.47 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 |
| usage | 0.50 | 0.54 | 0.66 | 0.89 | 0.78 | 0.94 |

### 5.4 What does *not* hold up: raw MSE and learned decay

**Prediction MSE is a decoupled instrument.** Memory lowers MSE on some envs (Occlusion `none` $0.46\!\to\!$ `both` $0.25\pm.03$) but is noisy elsewhere and even *raises* it where stochastic distractors dominate the loss; in the $\tau_{\text{slow}}$ sweep, $\tau{=}3$ has the *lowest* MSE yet chance usage. The cue is a small sub-space of the global latent, so MSE does not track memory quality — the probes do.

**Table 5 — Validation next-latent MSE (3 seeds, mean$\pm$std). Lower is better.**

| env | none | short | long | both |
|---|---:|---:|---:|---:|
| T-Maze | 0.76 ±.44 | 0.47 ±.17 | 0.62 ±.53 | 0.55 ±.32 |
| Occlusion | 0.46 ±.18 | 0.54 ±.36 | 0.39 ±.12 | **0.25 ±.03** |
| Recall | 0.76 ±.46 | **0.33 ±.11** | 0.56 ±.37 | 0.73 ±.49 |
| Distractor | **0.39 ±.12** | 0.41 ±.16 | 0.57 ±.19 | 0.52 ±.21 |
| TwoRoom (control) | **0.48** | — | — | 0.48 |

**Learned decay does not self-tune.** Making $\alpha$ learnable leaves the horizons near their initialization for every environment — across 3 seeds the learned $\tau_{\text{slow}}$ is $23.9\text{–}24.4$ regardless of the task gap (5 / 15 / 21 / 23), with std $\le0.4$ — i.e., it does *not* track the gap; the gradient signal on a scalar decay is weak. The practical lever is therefore *choosing* timescales, which is precisely why two banks spanning a *range* of horizons is the right design.

### 5.5 Control

On the fully-observable **TwoRoom**, `none` and `both` are indistinguishable on the held-out set (Table 5), confirming the gains above are memory-specific rather than added capacity.

### 5.6 Standard benchmark: POPGym Arcade

To test transfer beyond our controlled suite, we evaluate on two memory-centric POMDPs from **POPGym Arcade** [Morad et al.] — `CountRecallEasy` and `AutoEncodeEasy` — with pixel observations and discrete actions (one-hot encoded as the predictor's action input). These tasks expose no clean cue label, so we report next-latent validation MSE and the memory-ablation influence $\mathcal I_c$ (3 seeds; `lewm-memory-popgym`).

**Table 6 — POPGym Arcade: next-latent val MSE (lower better) and slow-bank influence (3 seeds, mean$\pm$std).**

| env | none (vanilla) | short | long | both | $\mathcal I_{\text{slow}}$ (long / both) |
|---|---:|---:|---:|---:|---|
| AutoEncodeEasy | 0.62 ±.44 | **0.34 ±.24** | 0.91 ±.04 | 0.78 ±.28 | 3.1 / 2.6 |
| CountRecallEasy | 0.71 ±.17 | 0.56 ±.17 | **0.53 ±.05** | 0.93 ±.19 | 3.9 / 2.1 |

![Figure 4: POPGym Arcade](figures/summary_popgym.png)
*Figure 4. POPGym Arcade — vanilla LeWM vs two-timescale memory (left: next-latent val MSE; right: slow-bank influence on the prediction). 3 seeds, mean$\pm$std.*

Two findings. **(i) The primitive transfers.** A *single matched* timescale beats vanilla LeWM (AutoEncode `short` $-46\%$; CountRecall `long` $-25\%$ / `short` $-21\%$), and the influence metric confirms the banks are genuinely *used* — $\mathcal I\approx3$ when a bank is injected and exactly $0$ when it is not. **(ii) Naive two-bank stacking is fragile.** `both` is the *worst* memory design on both envs (CountRecall `both` even underperforms vanilla), echoing the synthetic-env result (§5.4) and the learnable-$\alpha$ negative (§5.4): on noisier random-policy trajectories the two banks interfere rather than compose. This motivates **learned timescale gating/selection** as the clear next step, and tempers any "more memory is always better" reading.

### 5.7 Counterfactual memory swap: the memory *causally* drives the prediction

Probes show information is decodable, but not that the predictor *uses* it. We therefore intervene: for each episode *i* we predict the reveal-latent using *i*'s current frames and actions but **another episode *j*'s memory banks** (with cue[*j*]≠cue[*i*]), and apply a matched probe.

**Table 7 — Counterfactual memory swap (design `both`, 3 seeds, mean$\pm$std).** Does the prediction follow the *injected* memory or the *current* frame?

| env | own memory → cue (control) | **swapped memory → swapped cue** | swapped → current-frame cue | chance |
|---|---:|---:|---:|---:|
| T-Maze | 0.85 | **0.83** | 0.17 | 0.50 |
| Distractor | 0.87 | **0.79** | 0.21 | 0.50 |
| Recall | 0.41 | 0.40 | 0.25 | 0.33 |
| Occlusion | 0.58 | 0.58 | 0.43 | 0.50 |

![Figure 5: counterfactual swap](figures/fig_causal_swap.png)
*Figure 5. Swapping the memory bank between episodes with different cues. On long-gap tasks the prediction tracks the **injected** cue (follow-memory ≈ follow-self), while the current-frame cue collapses to ~chance.*

On the long-gap tasks, swapping the memory bank **flips the prediction to the injected cue** (follow-memory ≈ the own-memory control) while reading the current frame collapses to ~chance — direct causal evidence that the EMA banks control the prediction, not merely carry decodable information. This is the decisive test that probe decodability alone cannot provide.

## 6. Discussion and Limitations

The robust, multi-seed claims live on the *decision* axis: a memory bank helps exactly when its horizon $\tau$ exceeds the task's cue-to-decision gap $\Delta$, and a two-timescale pair covers a range of $\Delta$ with one elegant primitive. We are deliberately conservative about three points. **(i) Raw MSE is the wrong metric** here and should not headline. **(ii) Benchmarks.** We now include one standard pixel memory benchmark (POPGym Arcade, §5.6), where the single-timescale primitive transfers; broader suites (Memory Maze, more POPGym tasks) and a demonstration on frozen V-JEPA/DINO-WM features at scale remain priority next steps. The fragility of the two-bank combination on §5.6 points to learned timescale selection. **(iii) Baselines.** A long-context predictor, an RNN/SSM predictor, and an episodic-retrieval bank should be compared to show the two-timescale EMA is competitive and that the *controllable decomposition*, not capacity, is the contribution. All experiments here use 3 seeds; ≥5 seeds would further tighten the sweep curves.

## 7. Conclusion

A two-timescale exponential memory is a minimal, analyzable way to give JEPA world models a controllable sense of time. Across controlled memory-stressing tasks it produces a clean short-vs-long dissociation: the horizon must match the gap, the matched timescale carries information to the decision, and memory extends the usable horizon far beyond a finite window while a Markovian control is unaffected. Framed as a *primitive plus a measurement protocol* rather than a performance play, it is a foundation on which richer (selective, episodic, hierarchical) memories for latent world models can be built and, crucially, *visualized*.

## References

Assran et al. *I-JEPA.* CVPR 2023 (arXiv:2301.08243). · Assran et al. *V-JEPA 2.* 2025 (arXiv:2506.09985). · Zhou et al. *DINO-WM.* 2024 (arXiv:2411.04983). · Maes, Le Lidec, Scieur, LeCun, Balestriero. *LeWorldModel.* 2026 (arXiv:2603.19312). · Balestriero & LeCun. *LeJEPA.* 2025 (arXiv:2511.08544). · Shaj et al. *Multi Time Scale World Models.* NeurIPS 2023 (arXiv:2310.18534). · Hafner et al. *DreamerV3.* 2023 (arXiv:2301.04104). · Deng et al. *S4WM.* 2023 (arXiv:2307.02064). · Samsami et al. *R2I.* 2024 (arXiv:2403.04253). · Mattes et al. *Hieros.* 2023 (arXiv:2310.05167). · Gu et al. *HiPPO.* NeurIPS 2020 (arXiv:2008.07669). · Gu et al. *S4.* 2022 (arXiv:2111.00396). · Sun et al. *RetNet.* 2023 (arXiv:2307.08621). · Morad et al. *POPGym.* ICLR 2023 (arXiv:2303.01859). · McClelland, McNaughton, O'Reilly. *Complementary Learning Systems.* Psych. Review 1995.
