# Research Brief: Two-Timescale Memory for JEPA-Style World Models

*LeWorldModel-Memory — foundation document for an ICLR 2027 submission. Lead author: chrislin@nvidia.com. Date: 2026-06-25.*

---

## 1. Executive Summary

- **Problem.** JEPA-style world models (I-JEPA, V-JEPA 2, DINO-WM, PLDM, and the in-house LeWorldModel/"LeWM") have strong *per-frame* representations but **no persistent memory of how short- vs long-term context shapes the *dynamics* of the representation space**. Their temporal context is either nothing (I-JEPA), a fixed finite window (DINO-WM history length `H`, V-JEPA 2's ~16-frame block-causal window), or a single-timescale GRU (PLDM). All explicitly admit **autoregressive error accumulation** as the dominant long-horizon failure mode [Assran 2025; Sobal 2025; Terver 2026].
- **Gap.** The field's *current* answer to long-horizon failure is **hierarchy / subgoals** (FF-JEPA, HWM) [Masip 2026; Zhang 2026], **not memory**. An explicit, multi-timescale memory is therefore a genuinely distinct and underexplored axis — and LeCun's own world-model proposal lists a *separate* short-term memory module beside the predictor, framing memory as a missing component rather than an add-on.
- **Primary method.** A **two-timescale exponential-moving-average (EMA) memory**: two leaky integrators over the encoder latent stream, a *fast* bank (small `τ`, working memory) and a *slow* bank (large `τ`, long-term memory), additively injected into the predictor with **zero-initialized projections** so training begins *exactly* at the memoryless baseline. Implemented in `lewm/models/memory.py` + `lewm/models/memory_model.py`.
- **Elegance.** The whole mechanism is a diagonal linear state-space model (the simplest SSM). It adds **two scalars** (`α_fast`, `α_slow`) and **two `D×D` zero-init projections**; the training loss stays **2 terms, 1 λ** (`L_pred + λ·SIGReg`). The EMA decay kernel `K(k)=α(1−α)^k` has a closed-form **effective horizon `τ = −1/ln(1−α)`**, giving a directly *plottable* mathematical object that future work can use to **visualize** how each timescale shapes decisions.
- **Mathematical support.** We derive the kernel and horizon, and define a **decision-influence functional** — the L2 movement of the predicted latent (or planned action) when a memory bank is ablated — as the operational measure of "how memory affects the decision." Implemented as `MemoryLeWorldModel.memory_influence`.
- **Evaluation.** Four memory-stressing partially-observable envs (`tmaze`=long, `occlusion`=short, `recall`=mixed, `distractor`=long+interference) plus the fully-observable `tworoom` **control** where memory provably cannot help. Ablation matrix = {None, Short, Long, Short+Long} × 5 envs, one env per GPU. Probes decode the cue from each stream over time (availability) and from the model's *predicted* reveal latent (usage).
- **Why foundational.** It is the minimal, mathematically transparent primitive for putting controllable memory into the *dynamics* of any JEPA latent space, with a built-in visualization story (kernels, `τ` evolution, per-bank influence). It subsumes the existing finite-window as the `τ→0` limit and connects JEPA to SSMs (S4/Mamba), CLS neuroscience, and RSSM/Dreamer memory literature.

---

## 2. Related Work

### 2.1 JEPA & latent-space world models
JEPA models share one recipe: encode observations into latents and have a predictor forecast *future/masked latent representations* (not pixels) with anti-collapse machinery. **I-JEPA** [Assran 2023] is purely spatial — context encoder + EMA target encoder (`θ̄ ← τθ̄ + (1−τ)θ`, `τ`: 0.996→1.0) + narrow predictor; no temporal predictor, no memory. **V-JEPA 2 / V-JEPA 2-AC** [Assran 2025] extend to video at scale and autoregressively predict the next-frame latent under **block-causal attention** over a short window, planning with **MPC + CEM** (receding horizon). **DINO-WM** [Zhou 2024] builds a ViT transition model on frozen DINOv2 features conditioned on a **fixed history window `z_{t-H:t-1}, a_{t-H:t-1}`** with a causal mask; its Table 6 ablation (PushT, with mask) shows `h=1→0.76, h=2→0.88, h=3→0.92` — context helps, but only within the window. **PLDM** [Sobal 2025] uses a VICReg-derived multi-term anti-collapse loss and a **2-layer GRU** predictor (single-timescale), planning with MPPI; it explicitly lists accumulating long-horizon prediction errors as future work. *(Correction vs. our internal angle report: PLDM's objective has **six** terms — `L_sim + α·L_var + β·L_cov + λ·L_time-var + δ·L_time-sim + ω·L_IDM` — not five, and PLDM is reported ~**100×** slower than model-free, not ~4×.)*

- I-JEPA: https://arxiv.org/abs/2301.08243
- V-JEPA 2: https://arxiv.org/abs/2506.09985
- DINO-WM: https://arxiv.org/abs/2411.04983
- PLDM: https://arxiv.org/abs/2502.14819

The **failure mode** is verbatim in V-JEPA 2 §4.3 (Limitations): *"autoregressive prediction suffers from error accumulation: the accuracy of the representation-space predictions decreases with longer autoregressive rollouts,"* and the search space *"increases exponentially given a linear increase in the planning horizon."* [Terver 2026] formalizes this: compounding errors grow **exponentially with horizon `H`** when the predictor's Lipschitz constant `Λ ≥ 1`; multistep-rollout training trades higher one-step error `δ_K` for lower `Λ_K` (optimal `K=6` DROID, `K=2` sim). The frontier response is **hierarchy**, not memory: flat LeWM fails at random init (0.00%), **FF-JEPA** adds an action-free subgoal planner (`H=25`) lifting 75-step PushT 3.52%→91.80% and random-init 0.00%→82.42% [Masip 2026]; **HWM** reports up to +44% success and up to 4× cheaper planning [Zhang 2026]. *This hierarchy-vs-memory split is the core novelty wedge.*

- Terver 2026: https://arxiv.org/abs/2512.24497
- FF-JEPA: https://arxiv.org/abs/2606.09311 · HWM: https://arxiv.org/abs/2604.03208

### 2.2 RSSM / Dreamer / R2I / S4WM (model-based RL memory carriers)
**DreamerV3** [Hafner 2023] carries memory in a fixed-size RSSM model state `s_t=(h_t, z_t)`: a (LayerNorm-)GRU deterministic state `h_t = f_φ(h_{t-1}, z_{t-1}, a_{t-1})` plus a stochastic latent (32 categoricals × 32 classes); a recurrent bottleneck through which *all* past must pass — strong short-horizon, weak long-range. **TransDreamer/TSSM** [Chen 2021] swaps the GRU for attention over the full history — uncompressed memory, quadratic cost, wins on long-range memory tasks. **S4WM** [Deng 2023] and **R2I** [Samsami 2024] swap in **structured state-space models** (S4/S5): an SSM beats Transformers on long-range memory at sub-quadratic cost; R2I is superhuman on Memory Maze, SOTA on BSuite/POPGym, parity on Atari/DMC, faster than DreamerV3. **Hieros** [Mattes 2023] adds **hierarchical multi-timescale imagination** on an S5 backbone — fine timescales = short-term, coarse = long-term, directly analogous to our fast/slow split.

- DreamerV3: https://arxiv.org/abs/2301.04104 · TransDreamer: https://arxiv.org/abs/2202.09481
- S4WM: https://arxiv.org/abs/2307.02064 · R2I: https://arxiv.org/abs/2403.04253 · Hieros: https://arxiv.org/abs/2310.05167

### 2.3 SSM & linear-attention memory math
**HiPPO** [Gu 2020] frames memory as optimal projection of the past under a *measure*: LagT (exponential measure) → fixed-width sliding window; LegS (scaled measure) → timescale-robust all-history. **S4** [Gu 2022] is an LTI SSM `x'=Ax+Bu, y=Cx` whose memory *is* an exponential convolution kernel `K=(CB, CAB, CA²B, …)`; eigenvalues of `A` and step `Δ` set the horizon. **RetNet** [Sun 2023] gives the cleanest closed form: decay matrix `D_{nm}=γ^{n−m}`, recurrence `S_t = γ S_{t−1} + K_t^⊤V_t`, with **per-head multi-scale decay `γ_h = 1 − 2^{−5−h}`** (a spectrum of horizons). **GLA** [Yang 2024] makes decay **data-dependent**: `S_t = Diag(α_t) S_{t−1} + k_t^⊤v_t` (an input-conditioned EMA). **Mega** [Ma 2023] is the minimal version: a damped EMA `y_t = α x_t + (1−α) y_{t−1}` as an inductive bias inside gated attention. **Our two-timescale EMA is exactly two diagonal SSMs / two Mega EMAs at different `α`** — the simplest, most interpretable member of this family.

- HiPPO: https://arxiv.org/abs/2008.07669 · S4: https://arxiv.org/abs/2111.00396
- RetNet: https://arxiv.org/abs/2307.08621 · GLA: https://arxiv.org/abs/2312.06635 · Mega: https://arxiv.org/abs/2209.10655

### 2.4 CLS / neuroscience two-timescale grounding
**Complementary Learning Systems** [McClelland 1995; Kumaran 2016] derive *why* a fast/slow split is necessary: a single slow learner must interleave to capture statistics (avoiding catastrophic interference), so a fast store is needed to hold individual episodes immediately and consolidate gradually. The EMA consolidation `θ_slow ← (1−1/τ)θ_slow + (1/τ)θ_fast` is the clean ML analogue. ML instantiations: Dual Memory Networks [Kamra 2017], MHN+VAE pattern-separation/completion [Jun 2025], BrainCL wake/sleep consolidation [Liu 2025]. This answers the reviewer question *"why two timescales?"* with first-principles theory.

- CLS 1995: https://pubmed.ncbi.nlm.nih.gov/7624455/ · CLS 2016: https://www.cell.com/trends/cognitive-sciences/abstract/S1364-6613(16)30043-2

### 2.5 Episodic / retrieval memory
**Memorizing Transformers** [Wu 2022] add one kNN-augmented layer over a non-differentiable (k,v) bank fused via a learned per-head gate `g=σ(b_g)`, `V_a = V_m⊙g + V_c⊙(1−g)` (L2-normalize keys, no positional bias, FIFO, scales to 262K). **HCAM** [Lampinen 2021] does coarse-to-fine chunked retrieval. **RA-DT** [Schmied 2024] retrieves sub-trajectories with return-weighted utility scoring (but reports *no* in-context gain on Meta-World/Procgen — a cautionary negative result). [Pink 2025] argues episodic memory is the missing piece for long-term agents; [Omidi 2025] taxonomizes the design space. These define our **secondary "episodic bank" alternative** (§4).

- Memorizing Transformers: https://arxiv.org/abs/2203.08913 · HCAM: https://arxiv.org/abs/2105.14039 · RA-DT: https://arxiv.org/abs/2410.07071

### 2.6 Memory benchmarks & probing/visualizing representation dynamics
**Ni 2023** formally decouples **memory length** from **credit-assignment length** with the Passive/Active T-Maze (the dense shaping reward `R_t=(1[x_{t+1}≥t]−1)/(T−1)` makes credit-assignment length `c=1` while memory length = horizon) — the single most important design rule for our envs. **Cherepanov 2024** defines short- vs long-term via the **correlation horizon `ξ` vs context length `K`** (LTM iff `ξ > K`) and gives criteria to *prove* a benchmark isolates a memory type. **POPGym** [Morad 2023] (fast, 13 baselines incl. S4D/LMU, MMER metric), **Memory Gym** [Pleines 2024] (endless tasks; GRU beats Transformer-XL), and **Memory Maze** [Pasukonis 2022] (offline **MLP probe** on frozen representations decoding wall layout/object locations; Dreamer collapses 33.2→5.6 on large mazes) supply our probing protocol and motivation.

- Ni 2023: https://arxiv.org/abs/2307.03864 · Cherepanov 2024: https://arxiv.org/abs/2412.06531
- POPGym: https://arxiv.org/abs/2303.01859 · Memory Gym: https://arxiv.org/abs/2309.17207 · Memory Maze: https://arxiv.org/abs/2210.13383
- World-model memory survey: Laird & Clark 2025, https://arxiv.org/abs/2512.06983

---

## 3. The Gap: why JEPA world models have no memory, and what "memory shaping the dynamics" means

**Why the architecture forbids memory.** In LeWM (`lewm/models/leworldmodel.py`):
1. The encoder is **per-frame and memoryless**: `encode()` reshapes `(B,N,C,H,W)→(B·N,…)` and applies a ViT independently to each frame. `z_t = f_θ(o_t)` has *no* dependence on `o_{<t}`.
2. The predictor attends only over a window of `history_len=3` latents (`z_{t-H:t-1}, a_{t-H:t-1}`). Any correlation whose horizon `ξ` exceeds `H` is **structurally unrepresentable** — it must be re-derived from the current frame, which by partial observability is impossible once the cue leaves view.
3. Planning (`plan()`) rolls the predictor autoregressively in latent space with **no carried state** between steps, so memory cannot accumulate across the rollout.

Hence LeWM is, in the Efroni 2022 sense, only able to solve `m`-step-decodable tasks with `m ≤ H`; for any cue with `ξ > H` it is provably incapable, and on the fully-observable `tworoom` control (`ξ=0`) memory cannot help — the clean negative control.

**Formalizing "short- vs long-term memory affecting representation-space dynamics."** Let `z_t = f_θ(o_t)` be the memoryless latent and `g_φ` the predictor. The *dynamics* are the map `ẑ_{t+1} = g_φ(z_{t-H:t}, a_{t-H:t})`. We augment the predictor's input with a **memory state** `M_t = (m^f_t, m^s_t)` computed causally from the *entire* history `z_{≤t}`:

```
z̃_t = z_t + W_f m^f_t + W_s m^s_t        (fused latent; Eq. 4)
ẑ_{t+1} = g_φ(z̃_{t-H:t}, a_{t-H:t})
```

Memory **affects the dynamics** precisely when `∂ẑ_{t+1}/∂M_t ≠ 0`. We separate timescales by the correlation horizon `ξ` they can bridge (Cherepanov 2024): the fast bank carries information `ξ ≲ τ_fast` (short-term/working), the slow bank `ξ ≲ τ_slow` (long-term/episodic). The visualization claim — *future work can see how short- vs long-term memory shape decisions* — is then operationalized as two measurable objects: (A) **availability** = cue decodability from `m^f`/`m^s` vs time-since-cue; (B) **usage** = the causal influence `‖g_φ(M_t) − g_φ(M_t \ bank)‖` of each bank on the prediction/decision (§5, §6).

---

## 4. Proposed Method (PRIMARY: two-timescale EMA memory)

### 4.1 The two EMA banks
Over the SIGReg-regularized latent stream `z_t`, maintain two leaky integrators (`lewm/models/memory.py::TwoTimescaleMemory`):

```
m_t = (1 − α) m_{t-1} + α z_t                          (Eq. 1; one bank)
m^f_t = (1 − α_f) m^f_{t-1} + α_f z_t   (fast, large α_f → short τ_f)
m^s_t = (1 − α_s) m^s_{t-1} + α_s z_t   (slow, small α_s → long  τ_s)
```

Warm-started `m_0 = z_0` (unbiased early estimate). The recurrence unrolls into a **causal convolution with an exponential kernel**:

```
m_t = α Σ_{k≥0} (1−α)^k z_{t−k},   K(k) = α (1−α)^k            (Eq. 2)
```

with **effective horizon (time constant)**

```
τ = −1 / ln(1−α)   ( ≈ 1/α for small α )                       (Eq. 3)
```

`α` is parameterized through a logit (`α = σ(raw_α) ∈ (0,1)`), so it can be **fixed** (clean known horizons) or **learned** (discovered `τ` is logged via `horizons()`). The kernel is exposed (`kernel(length, which)`) for plotting.

### 4.2 Injection into the predictor (the only structural change)
`lewm/models/memory.py::MemoryFusion` applies Eq. 4 with **zero-initialized** `W_f, W_s` (mirroring AdaLN's zero-init). At step 0 the model is *exactly* memoryless LeWM and learns to recruit memory only as it helps. The `mode ∈ {none, short, long, both}` flag gives the **four ablations for free**:

```
none  → z̃ = z              short → z̃ = z + W_f m^f
long  → z̃ = z + W_s m^s    both  → z̃ = z + W_f m^f + W_s m^s
```

**Critical isolation trick** (`memory_model.py::compute_loss`): the predictor still sees only a window of `history_len` latents, but training uses a **sliding short window over a long chunk** (`unfold(1, h, 1)`). Any information that must travel further than `H` steps can *only* do so through the EMA banks — the memory is the **sole long-range channel**, so its contribution is causally identifiable.

### 4.3 Loss & parameter-count impact (the elegance argument)
Loss is **unchanged in form** — still 2 terms, 1 hyperparameter:

```
L = L_pred + λ · SIGReg(Z),   L_pred = ‖ẑ_last − z_{s+H}‖²
```

SIGReg already prevents latent collapse, so (unlike VICReg-based PLDM) **no extra variance/covariance term is needed** for the memory. Added parameters: `2` EMA logits + `2·D²` for `W_f, W_s` (e.g. `D=192` → `2·192² ≈ 73.7K` params, **<0.5%** of the ~15M LeWM). New knobs: `{τ_fast, τ_slow, learnable_alpha, memory_mode}` — all interpretable. This is the foundational/elegant criterion satisfied: minimal surface area, closed-form math, plottable internals.

### 4.4 Memory-aware rollout/planning
`rollout_latents()` seeds `(m^f, m^s)` from observed context, then advances the banks with each *predicted* latent via `memory.step()` during imagination — so memory persists across the autoregressive rollout (the channel that error-accumulation-bound JEPA lacks). This drops directly into the CEM planner.

### 4.5 Alternatives considered (and why EMA wins)

| Alternative | Mechanism | Pros | Cons vs. EMA |
|---|---|---|---|
| **Gated RNN latent state** (GRU, à la PLDM/RSSM) | `h_t = GRU(h_{t-1}, z_t, a_t)`, inject `h_t` | Learns content-dependent gating; strong short-horizon | **Single timescale**; no closed-form horizon to plot; gradients degrade long-range; ≫ params; not ablatable into clean short/long banks. EMA is the `α`-fixed, interpretable special case. |
| **Episodic retrieval bank** (Memorizing-Transformer/HCAM/RA-DT) | Store past `(k,v)` latents, kNN-retrieve, gate in | Exact distant recall; scales to very long horizons | Non-differentiable store, eviction/staleness, retrieval-quality dependence; RA-DT shows *no* in-context gain on some domains; far more moving parts; harder to visualize "the decay." |
| **Data-dependent decay** (GLA `Diag(α_t)`) | Input-conditioned EMA | More expressive | Loses the *fixed, known* `τ` that makes the visualization clean; a natural **follow-up**, not the foundational primitive. |

**EMA is preferred** because it is (i) a closed-form, plottable kernel with a single horizon scalar per bank, (ii) trivially ablatable into the exact {None/Short/Long/Both} matrix, (iii) <0.5% params with no new loss term, and (iv) the minimal common ancestor of SSMs, CLS consolidation, and RetNet/Mega — maximizing the foundational framing. The episodic bank is kept as the principal **scaling/exact-recall extension** for the discussion.

---

## 5. Mathematical Support

### 5.1 Kernel and effective horizon (derivation)
From Eq. 1, expand recursively:
`m_t = α z_t + (1−α)m_{t-1} = α z_t + (1−α)[α z_{t-1} + (1−α)m_{t-2}] = α Σ_{k=0}^{t} (1−α)^k z_{t−k}` (with `m_{-1}` absorbed by warm-start). Weights sum to ≈1: `α Σ_{k≥0}(1−α)^k = α·(1/α) = 1`. The kernel `K(k)=α(1−α)^k` decays geometrically; the lag where weight falls to `1/e` of `K(0)` satisfies `(1−α)^k = e^{-1}`, i.e. `k = −1/ln(1−α) = τ` (Eq. 3). Centroid (mean lag) `Σ k·K(k) = (1−α)/α ≈ τ` for small `α`, confirming `τ` is the memory's center of mass. Inverse map (`tau_to_alpha`): `α = 1 − e^{−1/τ}`.

**Connection to SSMs.** Eq. 1 is a diagonal LTI SSM with `A = (1−α)`, `B = α`, `C = 1` — kernel `K=(CB, CAB, …) = (α, α(1−α), α(1−α)², …)`, identical to Eq. 2. Two banks = a 2-channel diagonal SSM with two real poles `{1−α_f, 1−α_s}`. RetNet's per-head `γ_h = 1−2^{−5−h}` is the same construction with `γ = 1−α`; our two banks are a deliberate, interpretable 2-point sampling of that spectrum. This makes the method a *named special case* of the SSM family (defensible novelty: it is the first to use it as an explicit, ablatable memory inside a JEPA predictor with a visualization protocol).

### 5.2 Formalizing "how memory affects the decision"
Let the prediction functional be `Φ(M_t) = g_φ(z̃_{t-H:t}, a_{t-H:t})` with `z̃` depending on `M_t` through Eq. 4. Define the **causal decision-influence** of each bank by ablation (set its contribution to zero):

```
infl_fast(t) = ‖ Φ(m^f_t, m^s_t) − Φ(0, m^s_t) ‖₂
infl_slow(t) = ‖ Φ(m^f_t, m^s_t) − Φ(m^f_t, 0) ‖₂
```

Because `W_f, W_s` enter `z̃` linearly, this equals (to first order) the gradient-sensitivity `‖∂Φ/∂m·(W·m)‖`, so the ablation and gradient views agree. Implemented exactly in `memory_model.py::memory_influence` (returns `infl_fast`, `infl_slow` per episode). For **planning**, the analogous object is `Δa* = ‖a*(full) − a*(ablate bank)‖`, the shift in the CEM-optimal first action — the decision-level influence. A clean theoretical claim: on `tworoom` (ξ=0, Markovian) `infl_slow→0` at optimum (memory is provably useless), whereas on `tmaze` (ξ>H) `infl_slow > 0` is *required* for `cue_acc_from_prediction > chance` — a falsifiable, env-conditioned prediction.

### 5.3 Why two timescales reduce effective rollout depth
Following Terver 2026, autoregressive error grows like `Λ^H`. A slow bank supplies the cue-determined component of `ẑ` *directly* from history rather than through the chained predictor, so the predictor only needs to roll out the *short-horizon controllable* dynamics — reducing the effective number of error-compounding steps for the memory-determined sub-signal from `O(H)` to `O(1)`. This is the memory analogue of the hierarchy argument (FF-JEPA reduces rollout depth via subgoals; we reduce it via persistent state).

---

## 6. Experimental Design

### 6.1 Environments (`lewm/envs/memory_envs.py`)
Every env has a **cue-determined event** unrecoverable from the current frame, plus a controllable agent dot doing a random walk whose velocity is the 2-D action — so **actions drive the controllable part and memory drives the cue part** (clean separation). 64×64 RGB, `action_dim=2`, length 32.

| Env | Memory kind | `ξ` (cue→event) | Stresses |
|---|---|---|---|
| `tmaze` | **long** | cue off @3 → goal @24 (≈21) | slow bank |
| `occlusion` | **short** | hidden @11–17 (≈6) | fast bank (object permanence) |
| `recall` | **mixed** | shown @2–5 → replay @20 (≈15, ordered) | working + episodic |
| `distractor` | **long + interference** | cue @3 → goal @26 (≈23) with 5 random flashes | robust long-term recall |
| `tworoom` | **control (Markovian)** | ξ=0 | negative control — memory must *not* help |

These follow Ni 2023 (cue placement controls memory length independently of credit assignment) and Cherepanov 2024 (`ξ > H=3` for the long envs guarantees the task isolates long-term memory, not in-context shortcutting). The `info` dict ships `cue`, `cue_end`, `reveal`, `n_cue_classes` per episode for probing.

### 6.2 Ablation matrix (4 GPUs, one env per GPU)
`memory_mode ∈ {none, short, long, both}` × `{tmaze, occlusion, recall, distractor, tworoom}`. Secondary sweeps: `{τ_fast, τ_slow}` grid, `learnable_alpha ∈ {False, True}`, `history_len ∈ {1,3}`. Each cell × 3 seeds. (Launcher = Task #6.)

### 6.3 Metrics & visualizations (`lewm/eval/memory_probe.py`)
**(A) Availability — where the cue lives and for how long.** `probe_cue_over_time`: per-`t` multinomial logistic probe decoding the cue from each stream (`z`, `m^f`, `m^s`, `z̃`). Expected signature: `z` drops to chance the moment the cue leaves the frame; `m^f` decays over ≈`τ_fast`; `m^s` retains the cue out to the reveal step. → `plot_probe_curves` (accuracy vs time, with cue-off and reveal markers).

**(B) Usage — does the *decision* use it.** `decision_uses_memory`: train a probe on the *true* reveal latent, apply it to the model's *predicted* reveal latent. Metrics `cue_acc_from_prediction` vs `cue_acc_from_true_latent` vs chance. A Long/Both model's prediction encodes the cue-determined event (above chance); None/Short cannot at long `ξ`.

**(C) Causal influence.** `memory_influence` → `infl_fast`, `infl_slow` (mean L2 movement of the prediction under bank ablation). Plus `long_mem_advantage = acc(m^s) − acc(z)` and `short_mem_advantage` at reveal.

**(D) Kernel / horizon plots.** `plot_memory_kernels` draws `K(k)=α(1−α)^k` for both banks; learned `τ_fast/τ_slow` logged over training (kernel-shaping visualization).

**(E) Prediction & decision deltas.** `L_pred` at the reveal step (None vs Short vs Long vs Both); planning success / `Δa*` on goal-reaching; surprise / violation-of-expectation = prediction error spike at reveal for memoryless vs memory models (the memoryless model is "surprised" by the cue-determined event; the memory model is not).

**Predicted result table (qualitative):**

| | tworoom | occlusion (short) | tmaze (long) | distractor (long+int) | recall (mixed) |
|---|---|---|---|---|---|
| None | ✓ | ✗ | ✗ | ✗ | ✗ |
| Short | ✓ | ✓ | ~/✗ | ✗ | partial |
| Long | ✓ | ~ | ✓ | ✓ | partial |
| Short+Long | ✓ | ✓ | ✓ | ✓ | ✓ |

(Key falsifiable claim: **no memory mode beats None on `tworoom`** — if it does, the env is leaking and the result is confounded.)

---

## 7. ICLR 2027 Framing

**Contributions.**
1. **A minimal, closed-form two-timescale memory primitive** for JEPA world models: two EMA banks injected via zero-init projections, adding <0.5% params and **zero** new loss terms, with an analytic kernel `K(k)=α(1−α)^k` and horizon `τ=−1/ln(1−α)`.
2. **A formalization of memory acting on representation-space dynamics**, with an operational, gradient-consistent **decision-influence functional** that *visualizes* how short- vs long-term memory shape predictions and plans.
3. **A controlled benchmark suite** of four memory-stressing partially-observable envs spanning the short↔long axis plus a Markovian negative control, designed per the Ni-2023 memory/credit-assignment decoupling and the Cherepanov-2024 `ξ>K` criterion.
4. **A probing protocol** (availability + usage + causal influence) that yields publishable figures: probe-vs-time curves, kernel/`τ` plots, per-bank influence, and violation-of-expectation spikes.

**Why foundational.** It is the *simplest* mechanism that makes memory a first-class, controllable, *visualizable* property of a JEPA latent's dynamics; it subsumes the finite window (`τ→0`) and is the named diagonal-SSM/Mega special case, unifying JEPA with SSM, RSSM, and CLS literatures. It opens a different axis from the current frontier (hierarchy) and is composable with it.

**Expected results.** Clean monotone availability curves matching `τ`; Long/Both > Short > None on long-`ξ` envs; Short ≈ Both on `occlusion`; *all modes equal on `tworoom`*; positive `infl_slow` exactly where `ξ>H`; reduced violation-of-expectation at reveal for memory models.

**Risks & mitigations.**
- *Memory not recruited (W stays ≈0).* Verify gradient flow; warm-start `τ`; optionally add a small auxiliary cue-prediction probe loss — but only if needed (keep elegance).
- *SIGReg insufficient on memory streams.* Banks are convex averages of already-regularized `z`, so collapse risk is low; monitor per-dim variance of `m^s`.
- *Env leakage / shortcutting.* `tworoom` control + Cherepanov criterion guard against it; verify None fails on long envs.
- *Single-timescale baseline (GRU) competitive on short envs.* Expected and fine — the differentiator is long-`ξ` + interpretability; include the GRU as a baseline.
- *Reviewer "EMA is just an SSM."* Pre-empt: position as the *interpretable, ablatable, visualization-first* instantiation inside JEPA dynamics, with the decision-influence formalism as the novel analysis contribution.

---

## 8. Concrete Next Steps (quick validation on a simple env)

1. **Smoke test on `tmaze` (long), single GPU, `memory_mode=both`.** Generate episodes via `make_episode_fn('tmaze', img_size=64, length=32)`; train `MemoryLeWorldModel` with the sliding-window loss; confirm `loss/pred_loss/sigreg_loss` decrease and `memory.horizons()` logs sane `τ_fast≈2, τ_slow≈20`. (Wire into `scripts/train_memory.py`; finish Task #4 wandb logging, project `lewm-memory`.)
2. **Run `run_memory_eval` on a held-out batch** and inspect the two figures (`probe_cue_over_time`, `memory_kernels`). Success criterion: `acc_m_slow_at_reveal ≫ acc_z_at_reveal ≈ chance`, and `infl_slow > infl_fast` on `tmaze`.
3. **Negative-control check on `tworoom`:** train all four modes; confirm `long_mem_advantage ≈ 0` and `infl_slow ≈ 0` — memory provides no gain when fully observable.
4. **Two-mode contrast on `occlusion` vs `tmaze`:** show Short wins on `occlusion`, Long wins on `tmaze` (the fast/slow dissociation — the headline figure).
5. **`learnable_alpha=True` run:** verify the learned `τ_slow` drifts toward the env's `ξ` (memory adapts its horizon to the task) — a compelling "the model discovers the right timescale" result.
6. **Then scale out** via the 4-GPU launcher (Task #6: one env per GPU, full {None/Short/Long/Both} × 3 seeds) and assemble the ablation table + figures for the proposal doc (Task #7).

---

### Reference list (URLs)
I-JEPA https://arxiv.org/abs/2301.08243 · V-JEPA 2 https://arxiv.org/abs/2506.09985 · DINO-WM https://arxiv.org/abs/2411.04983 · PLDM https://arxiv.org/abs/2502.14819 · Terver 2026 https://arxiv.org/abs/2512.24497 · FF-JEPA https://arxiv.org/abs/2606.09311 · HWM https://arxiv.org/abs/2604.03208 · DreamerV3 https://arxiv.org/abs/2301.04104 · TransDreamer https://arxiv.org/abs/2202.09481 · S4WM https://arxiv.org/abs/2307.02064 · R2I https://arxiv.org/abs/2403.04253 · Hieros https://arxiv.org/abs/2310.05167 · HiPPO https://arxiv.org/abs/2008.07669 · S4 https://arxiv.org/abs/2111.00396 · RetNet https://arxiv.org/abs/2307.08621 · GLA https://arxiv.org/abs/2312.06635 · Mega https://arxiv.org/abs/2209.10655 · CLS 1995 https://pubmed.ncbi.nlm.nih.gov/7624455/ · CLS 2016 https://www.cell.com/trends/cognitive-sciences/abstract/S1364-6613(16)30043-2 · Memorizing Transformers https://arxiv.org/abs/2203.08913 · HCAM https://arxiv.org/abs/2105.14039 · RA-DT https://arxiv.org/abs/2410.07071 · Ni 2023 https://arxiv.org/abs/2307.03864 · Cherepanov 2024 https://arxiv.org/abs/2412.06531 · POPGym https://arxiv.org/abs/2303.01859 · Memory Gym https://arxiv.org/abs/2309.17207 · Memory Maze https://arxiv.org/abs/2210.13383 · Efroni 2022 https://arxiv.org/abs/2202.03983 · Laird & Clark 2025 https://arxiv.org/abs/2512.06983

**Key code anchors:** `lewm/models/memory.py` (banks, kernel, fusion), `lewm/models/memory_model.py` (sliding-window loss, `memory_influence`, `rollout_latents`), `lewm/envs/memory_envs.py` (5 envs), `lewm/eval/memory_probe.py` (availability/usage probes + figures).