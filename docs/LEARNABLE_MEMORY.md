# Selective Multi-Timescale Memory (SMT): a learnable, scalable short/long memory

*Proposal / design note. Branch: `learnable-memory`. Implementation: `lewm/models/memory.py::SelectiveMultiTimescaleMemory`, wired as `memory_impl='smt'` (`--memory-mode smt`).*

## 1. Motivation — what our study tells us to do next

The companion paper (`docs/ICLR.md`) establishes two empirical facts that, together, point directly at this design:

1. **A fixed log-spaced bank of EMA horizons is the best memory we tested** — it beats a learned GRU, a learned diagonal-SSM/RetNet-lite, and episodic retrieval on the long-gap tasks, *without any per-task tuning* (§5.11). *Spanning* horizons beats picking one.
2. **A learnable scalar decay does not self-tune** — making the EMA rate `α` learnable leaves the horizon stuck near its initialization regardless of the task gap (§5.4); the gradient signal on a raw decay rate is too weak.

The naive reading is "memory should be fixed." But fixed memory cannot *allocate* capacity, cannot be *input-dependent*, and does not obviously *scale* (it reads the same spectrum of horizons at every step for every input). The research question is therefore:

> **How do we make short/long memory learnable and scalable *without* re-introducing the thing that failed — learning the decay rates?**

**Answer (SMT):** keep the decays **fixed** (the reliable prior) and move **all** learnability to *input-conditioned gating* — a learned **write gate** (what to store) and a learned **read router** (which horizon to use, per step). Learning *selection over* a fixed timescale basis has a well-conditioned gradient (it is a function of the input, like attention), whereas learning the decay itself does not.

## 2. The architecture

![SMT architecture](figures/fig_smt_arch.png)
*Figure 1. SMT data flow. The latent `z_t` is gated by a learned **write gate** `i_t` and written into `K` **fixed** log-spaced EMA banks (`τ=2…64`); a learned **read router** `r_t` weights the banks; the weighted read-out is projected by `W_o` and added back residually. Blue = learned (`W_i,W_r,W_o`, ~1.5% of params); gray = fixed decays. All learnability is in the input-conditioned gating, never in the decay rates.*

Compact data flow:

```
                 ┌─ write gate  i_t = σ(W_i z_t) ─┐                    fixed banks
   z_t ──┬──────►│            (LEARNED)           │── i_t⊙z_t ──► [ τ=2 ][ τ=4 ]…[ τ=64 ]
         │       └───────────────────────────────┘                       │  │      │
         │                                                                ▼  ▼      ▼
         │        read router  r_t = g(W_r z_t)  (LEARNED) ───► weights r_{t,1..K}
         │                                                                │
         │                                          o_t = W_o( Σ_k r_{t,k} m^k_t )  (LEARNED)
         └──────────────────────────────── + ◄─────────────────────────────┘
                                            │
                                      z̃_t = z_t + o_t   ──►  predictor
```

Let `z_t ∈ R^D` be the encoder latent. SMT maintains `K` EMA banks at **fixed** log-spaced horizons `τ_1<…<τ_K` (default `τ ∈ {2,4,8,16,32,64}`), `a_k = 1 − e^{−1/τ_k}`:

```
write / input gate     i_t   = σ(W_i z_t)                       ∈ (0,1)^D     (what to store)
bank-k recurrence      m^k_t = (1 − a_k) m^k_{t−1} + a_k (i_t ⊙ z_t)         (a_k FIXED)
read router            r_t   = softmax(W_r z_t / T)             ∈ Δ^{K−1}     (which horizon)
memory read-out        o_t   = W_o ( Σ_k r_{t,k} · m^k_t )
injected latent        z̃_t   = z_t + o_t
```

Only `W_i` (D×D), `W_r` (D×K) and `W_o` (D×D) are learned — about `2D² + DK` parameters (~1.5% of the model). The decays `a_k` are buffers. Three design choices matter:

- **Fixed basis, learned selection.** The model never learns a timescale; it learns *which* of the known timescales to read and *what* to write into them. This sidesteps the weak-decay-gradient failure of §5.4 while keeping the spanning-horizons prior of §5.11.
- **Input-conditioned write gate** `i_t`. Mamba-style selectivity, but on *what to store* rather than *how fast to forget*: the model can ignore distractors and write only decision-relevant content into the (fixed-horizon) banks.
- **Small (not zero) read-out init.** The EMA/`multi` designs zero-init their read-outs to start exactly at the memoryless baseline. SMT cannot: the router and write gate sit *upstream* of the multiplicative read-out, so a zero read-out gives them *exactly zero gradient at step 0*. We instead use a small read-out init (≈5% deviation from baseline) so every learned part trains from the first step. *(Verified: with zero-init the router/gate gradients are 0; with small init they are non-zero.)*

## 3. Why it is scalable

- **Linear time / memory.** Each bank is a diagonal linear recurrence: `O(L·K·D)` with `K` small (log-spacing covers `τ∈[2,256]` with `K=8`). No `O(L²)` attention.
- **Parallelizable.** The recurrence `m^k_t = (1−a_k)m^k_{t−1} + a_k u_t` is an associative scan (prefix-sum form), so it parallelizes across the sequence exactly like S5/Mamba (the current code uses a simple sequential scan for the short `L=32` chunks; the parallel scan is a drop-in for long sequences).
- **Stackable → hierarchy by depth.** SMT is a layer. Stacking it (as HGRN2 stacks gated recurrences) yields a depth hierarchy of timescales on top of the within-layer spectrum — a route to deeper memory without growing `K`.
- **Constant state.** State is `K·D` regardless of sequence length (unlike retrieval/attention caches that grow with `L`).

## 4. Relation to prior work (and what is new)

| Method | Decay/timescale | Input-dependent? | Our difference |
|---|---|---|---|
| **Mamba / S6** (Gu & Dao 2023) | learned input-dependent `Δ` | yes (learns timescale) | we *fix* the timescale basis and learn the *selection over* it — learning `Δ` is the very thing that failed to self-tune in our regime (§5.4) |
| **Mega** (arXiv:2209.10655) | learned multi-dim EMA coefficients | no (static EMA) | we fix the EMA coefficients; learnability is in per-step routing + write gating |
| **HGRN2** (arXiv:2404.07904) | learned data-dependent decay, monotone by depth | yes | decays fixed; hierarchy optional via stacking, not via learned decay bounds |
| **RetNet** (Sun et al. 2023) | fixed multi-scale decay per head | no | we add input-conditioned read routing + write gating over the fixed scales |
| **Titans** (arXiv:2501.00663) | deep memory meta-learned at test time | yes (test-time) | orthogonal axis; SMT is a cheap, interpretable train-time module (composable with it) |
| **Instance-conditional timescales of decay** (arXiv:2212.05908) | mixture over fixed decay rates via a learned scorer | yes | closest idea, but for *non-stationary supervised instance weighting* — we bring it to *sequence/world-model memory* with per-step routing, a learned write gate, and the short/long interpretability protocol |

**Net positioning.** The literature either *learns the decays* (Mamba, Mega, HGRN2) — unreliable here — or uses *fixed multi-scale decays without selectivity* (RetNet). SMT is the missing quadrant: **fixed decay basis + learned input-conditioned selectivity (write + read)**, motivated by a controlled finding that this is exactly the split that works. To our knowledge this specific combination has not been proposed as a sequence-memory module for (JEPA) world models.

## 5. Interpretability — and an honest negative finding about the router

Because the horizons are fixed and known, the router output `r_t` is directly plottable as a **read preference over known horizons** (`route_weights()`), a learned analogue of the short-vs-long dissociation (§5.1). We visualize the trained `smt` (sigmoid/v2) router on three envs:

![SMT router](figures/fig_smt_router.png)
*Figure 2. SMT learned read router. (a) per-horizon mean read weight ≈ uniform (1/6) with only tiny, weakly task-appropriate deviations (Distractor leans slightly toward the longest τ=64). (b) the temporal std of the router is ~10⁻⁵ — i.e. essentially **constant over time**.*

**Honest finding: the router collapsed to a near-static, input-independent mixture.** On these clean tasks the trained read router barely deviates from a uniform, time-constant weighting (`W_r ≈ 0`; only its bias matters). This *explains why SMT-v2 matches `multi`*: with near-uniform additive gates it effectively **becomes a static additive K-bank with learned constant weights** — i.e. it recovered `multi`'s read-out rather than learning per-step content routing. The hypothesized selectivity *did not emerge* — the same weak-gradient degeneracy we saw for learned decay (§5.4), now for the router: when a static mixture already solves an easy task, there is no pressure to be input-dependent.

This sharpens the proposal's central question and motivates the next experiments (§6–§7): does input-dependent routing (and the write gate) *emerge and help* when a static mixture is **not** sufficient — i.e. on harder, interference-heavy tasks? (It also suggests an architectural lever: a router temperature / entropy or load-balancing pressure to encourage peakier, content-dependent routing.)

## 6. Experiment plan

1. **Headline comparison** — `none` vs fixed `multi` vs **`smt`** on the four memory envs × 3–5 seeds (usage, availability, influence). Hypothesis: SMT matches `multi` on the clean long-gap tasks and *beats* it on **Distractor** (write gate should suppress distractor flashes) and **Recall** (router should switch horizons across the sequence).
2. **Scalability** — longer chunks `L ∈ {32,64,128}` and the parallel-scan path; SMT vs `multi` vs a long-context predictor in time/quality.
3. **Selectivity ablations** — write-gate-only, router-only, both; vs Mamba-style learned-`Δ` under matched budget (does fixed-basis+selection beat learned-`Δ`, as §5.4 predicts?).
4. **Router visualization** — per-step `r_t` over the cue→decision gap and over the dm_control/OGBench occlusion rollouts (does the model route to long memory exactly across the blackout?).
5. **Real robots** — `smt` on the dm_control/OGBench occlusion suite (§5.15–5.16): does learned write-gating improve post-occlusion prediction over the fixed K-bank?

## 7. Initial validation results

**v1 (softmax mixture router), 4 envs × 3 seeds, 30 epochs (usage = cue decodable from the prediction; mean±std).**

| env | none | **multi (fixed)** | **smt (learnable)** | chance |
|---|---:|---:|---:|---:|
| T-Maze (Δ21) | 0.49 ±.02 | **0.99 ±.00** | 0.80 ±.06 | 0.50 |
| Distractor (Δ23) | 0.55 ±.03 | **1.00 ±.00** | 0.79 ±.04 | 0.50 |
| Recall (Δ15) | 0.32 ±.03 | **0.47 ±.01** | 0.40 ±.03 | 0.33 |
| Occlusion (Δ5) | 0.48 ±.06 | **0.71 ±.02** | 0.59 ±.02 | 0.50 |

Two honest takeaways:
1. **SMT is the strongest *learnable* memory so far.** On T-Maze it reaches 0.80, clearly above the paper's learned baselines (GRU 0.54, SSM 0.58, retrieval 0.72) and approaching the fixed-EMA `both` (0.84) — learning *selection over a fixed basis* is a real improvement over learning the dynamics.
2. **But it does not yet beat the fixed K-bank** (0.80 vs 0.99; 0.79 vs 1.00). The "fixed structure is a remarkably strong prior" thesis (§5.11) survives this stronger learnable challenger.

**Diagnosis.** `multi` reads *all* banks **additively** (each fully contributes via its own read-out); SMT-v1's **softmax** router is a *convex mixture*, so reading the decisive long bank requires *down-weighting* the others, attenuating exactly the signal that matters. This predicts a fix: replace the softmax mixture with **independent additive sigmoid gates** (every fixed-horizon bank can contribute fully, but input-conditioned) — i.e. `multi`'s additive read-out made content-selective.

**v2 (additive sigmoid gates, `--smt-router sigmoid`) — the diagnosis was correct.** Replacing the convex softmax mixture with independent input-conditioned sigmoid gates (every fixed-horizon bank can contribute fully) closes most of the gap to the fixed K-bank, 4 envs × 3 seeds:

| env | none | **multi (fixed)** | smt-v1 (softmax) | **smt-v2 (sigmoid)** | chance |
|---|---:|---:|---:|---:|---:|
| T-Maze (Δ21) | 0.49 | 0.99 | 0.80 | **0.96 ±.02** | 0.50 |
| Distractor (Δ23) | 0.55 | 1.00 | 0.79 | **0.97 ±.03** | 0.50 |
| Recall (Δ15) | 0.32 | 0.47 | 0.40 | 0.42 ±.03 | 0.33 |
| Occlusion (Δ5) | 0.48 | 0.71 | 0.59 | 0.61 ±.04 | 0.50 |

The change is large exactly where predicted — **+0.16 on T-Maze and +0.18 on Distractor** — confirming that the softmax mixture was the bottleneck (it down-weighted the decisive long bank). **SMT-v2 now matches the fixed K-bank on the clean long-gap tasks (0.96 vs 0.99; 0.97 vs 1.00) while being fully learnable, input-conditioned, and interpretable.** On the harder Recall (3-way) and short-gap Occlusion it narrows the gap but still trails slightly.

### 7.1 Real-robot SMT (exp #5)

On the cached dm_control + OGBench occlusion data, learnable `smt` (sigmoid) vs the fixed `multi` (next-latent val MSE, lower = better, 3 seeds):

| env (occluded) | none | **multi (fixed)** | **smt (learnable)** |
|---|---:|---:|---:|
| Reacher | 0.328 | **0.175** | 0.182 |
| Ball-in-Cup | 0.331 | 0.173 | **0.135** |
| Finger-spin | 0.333 | **0.175** | 0.176 |
| Cheetah | 0.240 | **0.201** | 0.227 |
| OGBench-Cube | 0.329 | **0.177** | 0.178 |

SMT generalizes to real robots and **matches `multi`** — one clear win (Ball-in-Cup, −22% vs multi, where selectively storing the ball's trajectory through the blackout helps), otherwise ties/slight losses; all far below memoryless `none`.

### 7.2 Selectivity under harder interference (exp #1, harder variants)

The decisive test set up by §5: does the learnable gating *beat* the fixed K-bank once a static mixture is no longer sufficient? Harder Distractor (more interference flashes) and Recall (longer sequence), usage (mean±std; chance Distractor 0.50, Recall 0.33):

| task (hardness) | none | **multi (fixed)** | **smt (learnable)** | smt − multi |
|---|---:|---:|---:|---:|
| Distractor n=10 | 0.44 | **0.98** | 0.96 | −0.02 |
| Distractor n=16 (harder) | 0.52 | **0.99** | 0.91 | **−0.08** |
| Recall seq=5 | 0.39 | **0.40** | 0.36 | −0.05 |
| Recall seq=7 (harder) | 0.32 | 0.36 | **0.42** | **+0.06** |

**The selectivity hypothesis is largely not confirmed.** Under *heavier* distractor interference SMT does **not** beat `multi`; the gap actually *widens against* SMT at n=16 (−0.08) — the opposite of the predicted "write gate suppresses distractors" effect. The lone positive is harder Recall (seq=7, +0.06), but in a regime where every method sits near 3-way chance. This is fully coherent with the router-collapse finding (§5): SMT's learnable selectivity does not strongly activate, so the extra machinery matches `multi` at best and slightly underperforms it under heavy interference (harder optimization, no selectivity payoff).

### 7.3 Conclusion (honest)

A learnable short/long memory can **match** the strong fixed K-bank — on clean tasks (§7 v2), on real robots (1 win / 2 ties / 2 slight losses, §7.1), and under moderate interference — *provided* learnability is placed on input-conditioned gating over a fixed timescale basis, not on the decays. But across every setting it does **not yet beat** the fixed prior, and under heavy interference it slightly trails (§7.2). The diagnosis is consistent end to end: the learned router collapses to a near-static, input-independent mixture (§5), so SMT mostly **recovers `multi`'s additive K-bank** rather than exploiting content-dependent selectivity. The fixed log-spaced K-bank thus **remains the prior to beat** — a striking reaffirmation of the paper's thesis (§5.11), now against a strong learnable challenger.

This sharpens the real open problem into two levers: **(i) make selectivity actually emerge** (sparsity/entropy or load-balancing pressure on the gates, temperature, or curriculum/harder training so a static mixture cannot win); and **(ii) relax the constant-size constraint** — replace the fixed K=6 bank with a **learnable, variable-size** multi-scale bank (an over-complete fixed-decay basis with *sparse, learned* selection to an emergent active set), so capacity is allocated by content rather than hand-set. That second lever is exactly the question explored in §8.

## References

Gu & Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* 2023 (arXiv:2312.00752). · Ma et al. *Mega: Moving Average Equipped Gated Attention.* 2022 (arXiv:2209.10655). · Qin et al. *HGRN2: Gated Linear RNNs with State Expansion.* 2024 (arXiv:2404.07904). · Sun et al. *Retentive Network (RetNet).* 2023 (arXiv:2307.08621). · Behrouz et al. *Titans: Learning to Memorize at Test Time.* 2025 (arXiv:2501.00663). · *Instance-Conditional Timescales of Decay for Non-Stationary Learning.* (arXiv:2212.05908). · Yang et al. *Gated Linear Attention.* 2023 (arXiv:2312.06635). · Companion paper: `docs/ICLR.md` (§5.4 learned-decay does not self-tune; §5.11 fixed K-bank beats learned memories).
