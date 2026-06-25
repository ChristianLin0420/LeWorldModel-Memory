# Two-Timescale Memory for JEPA-Style World Models

**A foundational study of how short- and long-term memory shape the dynamics of a learned representation space.**

*LeWorldModel-Memory — proposal for ICLR 2027. Date: 2026-06-25.*
*Companion literature review: [`RESEARCH_BRIEF.md`](RESEARCH_BRIEF.md). Implementation: `lewm/models/memory.py`, `lewm/models/memory_model.py`, `lewm/envs/memory_envs.py`, `lewm/eval/memory_probe.py`.*

---

## 1. One-paragraph thesis

JEPA-style world models (I-JEPA, V-JEPA 2, DINO-WM, PLDM, and the LeWorldModel/"LeWM" baseline in this repo) learn excellent **per-frame** representations, but their *temporal* context is either nothing, a **fixed finite window** (DINO-WM's history `H`, V-JEPA 2's block-causal window), or a **single-timescale** recurrent state (PLDM's GRU). They therefore have no controllable way to represent how information at *different temporal distances* shapes the dynamics of the latent space — and all of them report **autoregressive error accumulation** as the dominant long-horizon failure mode [Assran 2025; Sobal 2025]. The field's current answer to this is **hierarchy/subgoals** (FF-JEPA, HWM), *not memory*. We propose the minimal, mathematically transparent alternative: equip the predictor with a **two-timescale exponential memory** — a *fast* (short-term) and a *slow* (long-term) leaky integrator over the latent stream — and show, with controlled environments and probes, exactly **how short- vs long-term memory affect the model's decisions**.

## 2. Why this is foundational (and elegant)

- **Minimal.** The mechanism is two scalars (`α_fast`, `α_slow`) plus two zero-initialized `D×D` projections. The training loss is unchanged: `L = L_pred + λ·SIGReg(Z)` — still **2 terms, 1 hyperparameter**.
- **Closed-form & visualizable.** Memory = a diagonal linear state-space model whose kernel is an explicit exponential with a *plottable* effective horizon `τ`. The paper's central figures are mathematical objects, not black boxes.
- **A bridge.** It is the simplest member of the SSM family (S4/Mamba), the scalar case of RetNet's per-head multi-scale decay `γ_h`, the discrete analogue of HiPPO-LagT, and the algorithmic core of Complementary Learning Systems (fast hippocampus / slow neocortex). It subsumes the finite-window baseline as the `τ→0` limit.
- **A measurement, not just a model.** We define a *decision-influence functional* and a *probe-over-time* protocol that future work can reuse to audit memory in **any** JEPA latent space.

## 3. Method

### 3.1 Setup (the baseline we extend)
The encoder `E_θ` maps each frame independently to a latent `z_t = E_θ(o_t) ∈ ℝ^D` (memoryless; SIGReg keeps `z` ≈ isotropic Gaussian). The predictor `P_φ` attends over a short window of the last `h = history_len` latents plus actions and predicts the next latent. With no memory, anything older than `h` steps is unrecoverable.

### 3.2 Two-timescale EMA memory
We run two exponential-moving-average (leaky-integrator) banks over the latent stream:

$$ m^{(c)}_t = (1-\alpha_c)\, m^{(c)}_{t-1} + \alpha_c\, z_t, \qquad c \in \{\text{fast},\text{slow}\}. \tag{1}$$

Unrolling (1) shows the memory is a **causal convolution of the past with an exponential kernel**:

$$ m^{(c)}_t = \alpha_c \sum_{k\ge 0} (1-\alpha_c)^k\, z_{t-k}, \qquad K_c(k) = \alpha_c (1-\alpha_c)^k. \tag{2}$$

The kernel decays geometrically, giving a closed-form **effective memory horizon (time constant)**

$$ \tau_c = \frac{-1}{\ln(1-\alpha_c)} \;\approx\; \frac{1}{\alpha_c}\ \text{(small }\alpha\text{)}, \qquad \alpha_c = 1 - e^{-1/\tau_c}. \tag{3}$$

A **fast** bank (large `α`, small `τ`) is working/short-term memory; a **slow** bank (small `α`, large `τ`) is long-term memory. (Implemented and unit-tested against the closed form in `lewm/models/memory.py`; `α` is parameterized via a logit so `α∈(0,1)` always, and may be fixed for known horizons or learned.)

### 3.3 Injection (zero-init) and the four ablations
The banks are injected additively into the only thing the predictor ever sees:

$$ \tilde z_t = z_t + \mathbb{1}[\text{short}]\,W_f\, m^{(f)}_t + \mathbb{1}[\text{long}]\,W_s\, m^{(s)}_t. \tag{4}$$

`W_f, W_s` are **zero-initialized**, so training *begins exactly at the memoryless baseline* and recruits memory only as it reduces prediction loss — the same philosophy as the predictor's zero-init AdaLN. The indicator flags give the four ablations for free:

| design | `z̃` | memory available |
|---|---|---|
| `none` | `z` | — (baseline JEPA) |
| `short` | `z + W_f m^f` | fast only |
| `long` | `z + W_s m^s` | slow only |
| `both` | `z + W_f m^f + W_s m^s` | fast + slow |

### 3.4 Keeping the predictor short-context (the key control)
We train the predictor with a **sliding window of length `h`** over a longer chunk of length `L ≫ h` (`lewm/models/memory_model.py::compute_loss`). Each window predicts only its next latent. Because no window is longer than `h`, **information that must travel further than `h` steps can only do so through the EMA banks** — the memory is the *sole* long-range channel. This isolates the memory's contribution and makes the ablations clean.

### 3.5 Cost
Two scalars + two `D×D` matrices (`≈ 2D²` params; for `D=128`, ~33K — about 1.5% of the model) and an `O(L·D)` scan. No EMA target encoder, no stop-gradient, no reconstruction loss — the LeWM design constraints are preserved.

## 4. Mathematical support: formalizing "how memory affects the decision"

**Correlation horizon.** For a task whose decision at time `t` depends on a cue at `t-Δ`, the cue's weight in bank `c` is `K_c(Δ) = α_c(1-α_c)^Δ`. A bank can support the decision iff `K_c(Δ)` is non-negligible, i.e. roughly `Δ ≲ τ_c`. This predicts a **clean dissociation**: with `τ_fast = 3, τ_slow = 25`, a Δ≈5 gap (occlusion) is within reach of the fast bank, while a Δ≈21 gap (T-maze) is not — only the slow bank survives (`K_fast(21) ≈ 0.717^{21} ≈ 9·10^{-4}`). Short memory handles short gaps and **provably fails** long ones; long memory handles both.

**Decision-influence functional.** Let `f` be the predictor's imagined next latent (the decision proxy; for planning, the CEM-selected action). The **causal influence** of bank `c` on the decision is the movement induced by ablating it:

$$ \mathcal{I}_c \;=\; \big\| f(\tilde z) - f(\tilde z \,|\, W_c \leftarrow 0) \big\|_2. \tag{5}$$

This is the operational measure of "how much the decision uses memory `c`," computed in `MemoryLeWorldModel.memory_influence`. (Equivalently, the gradient `∂f/∂m^{(c)}` gives an infinitesimal version.) We report `I_fast`, `I_slow` per environment and per design.

## 5. Experiments

### 5.1 Environments (one per GPU) + control
Each env contains a **cue-determined event**: something appearing *later* is decided by a cue shown *earlier* and is not recoverable from the current frame or action — so a memoryless model cannot predict it. An independent random-walk agent dot supplies genuine action-conditioned dynamics orthogonal to the memory channel. (`lewm/envs/memory_envs.py`.)

| env | memory kind | Δ (cue→decision) | what must be remembered |
|---|---|---|---|
| `tmaze` | **long** | ~21 | which arm the early cue selected |
| `occlusion` | **short** | ~5 | the target's lane while it is briefly hidden |
| `recall` | **mixed** | ~15 | a short colour sequence, replayed after a delay |
| `distractor` | **long + interference** | ~23 | the *first* cue, despite random distractor flashes |
| `tworoom` | **Markovian control** | 0 | nothing (memory must give **no** advantage) |

### 5.2 Ablation matrix
`{none, short, long, both} × {5 envs}`, fixed horizons `τ=(3,25)`, one env per GPU, four designs sequentially (`scripts/run_all.sh`). Logged to wandb project **`lewm-memory`** (group = env, job_type = design, tags `env:*`, `design:*`, `kind:*`).

### 5.3 Two complementary measurements
- **Availability** (`probe_cue_over_time`): a linear probe decodes the cue from each stream (`z`, `m^f`, `m^s`) at every step. Expectation: `z` forgets the cue the instant it leaves frame; `m^f` holds it ~`τ_fast` steps; `m^s` holds it ~`τ_slow` steps. *This is the empirical signature of the kernel (2).* (Already visible at 2 epochs on `tmaze`: at the decision step `acc(z)=0.51`, `acc(m^f)=0.33`, `acc(m^s)=1.00`.)
- **Usage** (`decision_uses_memory` + influence (5)): train a probe on the *true* reveal latent, apply it to the model's *predicted* reveal latent. Only designs that **inject** the relevant bank make the cue decodable from the prediction. Availability is a property of the math (design-independent); **usage** is what the architecture buys.

### 5.4 Headline predictions
1. `tworoom`: all four designs equal (memory gives no advantage) — sanity that the gains are real.
2. `occlusion`: `short`, `long`, `both` all succeed; `none` fails (gap > `h`).
3. `tmaze`, `distractor`: `long`/`both` succeed; `short`/`none` fail (gap ≫ `τ_fast`).
4. Influence `I_slow` large exactly on the long-horizon envs; `I_fast` large on the short ones — a quantitative short↔long dissociation.

## 6. Contributions

1. **Reframing.** Position explicit multi-timescale *memory* (vs. the field's hierarchy/subgoals) as a distinct, underexplored axis for JEPA world models, and connect it to SSM/HiPPO/RetNet and CLS theory.
2. **Method.** A minimal two-timescale EMA memory with closed-form horizon and zero-init injection that preserves LeWM's 2-term loss.
3. **Measurement.** A reusable protocol — probe-over-time (availability) + decision-influence functional (usage) — for auditing memory in any latent world model, with a controlled env suite + Markovian control.
4. **Evidence.** A clean, mathematically predicted short↔long dissociation across the env suite.

## 7. Risks, limitations, alternatives
- **Encoder must encode the cue at cue-time** for EMA to carry it; if not, add a light auxiliary or rely on prediction pressure (zero-init lets the model learn it). Probing `z` at cue-time checks this.
- **Linear EMA is content-agnostic** (it cannot *select* what to store). We discuss two principled upgrades as ablations, both supersets of (1): a **learnable/input-dependent `α`** (→ Mamba-style selective SSM) and an **episodic retrieval bank** (attention over stored latents, à la Memorizing Transformers) for content-addressable long-term recall. The thesis is that the *fixed* two-timescale primitive is the right **foundational** baseline precisely because it is analyzable.
- **Scale.** Validated here on controlled pixel envs; scaling to V-JEPA-scale video is future work (the mechanism is `O(L·D)` and drop-in).

## 8. Reproduce
```bash
# full matrix across 4 GPUs (wandb project: lewm-memory)
EPOCHS=30 NUM_EPISODES=5000 bash scripts/run_all.sh
# single cell
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train_memory.py --env tmaze --memory-mode both --fixed-alpha --tau-fast 3 --tau-slow 25
# summary table + figure
.venv/bin/python scripts/aggregate_results.py
```

## Key references
JEPA/world models: I-JEPA (Assran 2023, arXiv:2301.08243), V-JEPA 2 (Assran 2025, arXiv:2506.09985), DINO-WM (Zhou 2024, arXiv:2411.04983), PLDM (Sobal 2025, arXiv:2502.14819). Long-horizon failure & hierarchy: FF-JEPA (Masip 2026), HWM (Zhang 2026). Memory carriers: DreamerV3 (Hafner 2023, arXiv:2301.04104), S4WM (Deng 2023, arXiv:2307.02064), R2I (Samsami 2024, arXiv:2403.04253), Hieros (Mattes 2023, arXiv:2310.05167). Memory math: HiPPO (Gu 2020, arXiv:2008.07669), S4 (Gu 2022, arXiv:2111.00396), RetNet (Sun 2023, arXiv:2307.08621). Memory eval: Ni et al. (NeurIPS 2023, arXiv:2307.03864). CLS: McClelland, McNaughton & O'Reilly (1995). *(Full annotated bibliography in `RESEARCH_BRIEF.md`.)*
