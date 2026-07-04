# V21 Proposal — Give the Belief a Consumer

**Status: PROPOSAL (awaiting review). Nothing here is implemented. Successor to the completed V20 program (`docs/V20_PROPOSAL.md` §8–§12: one Holm survivor, one inverted Tier-1, three falsified claims, two new instruments).**

This document does three things: (1) answers the submission question — is V20 an ICLR paper? — with an adversarial four-lens review panel run over the actual program documents, not optimism; (2) itemizes the V20 issues that panel converged on; (3) proposes V21 as the program that retires those objections in leverage order, built around one kernel idea the V20 evidence itself forces: **a belief is only as good as the decisions it licenses — the evaluation coordinate must move from probe decodability to certified-memory return, so the belief's uncertainty finally has a consumer.**

One sentence: **V20 built and validated the filter; V21 gives its output a buyer, gives its headline its own registration, and gives its claims an outside world.**

---

## 1. The submission question, answered adversarially

A simulated ICLR panel (four independent reviewer lenses — significance, rigor/external validity, claims–evidence match, positioning/baselines — plus an area-chair synthesis) was run over `docs/V20_PROPOSAL.md` §8–§12 and `docs/V19_PROPOSAL.md` §7–§12 with full repo access.

**Verdict: not submittable as-is.** Scores 5/4/5/4; area-chair estimate **reject, P(accept) ≈ 0.10** for the architecture-headline framing; ≈ 0.35 after reframing alone; **7–8 territory after the ranked fixes below.** All four lenses independently praised the same things (preregistration discipline, fail-closed gates, the certification methodology, the s\* instrument, honest negative results — "top-percentile experimental hygiene") and independently struck the same defect:

> **The headline claim outruns its evidence on four axes at once.** "A derived Kalman carrier with slow trust beats learned recurrence on certified-memory tasks" is (a) *epistemically mislabeled* — the +0.1225/p≈10⁻⁵ contrast is the quantification of a registered moot clause, not a Holm family member, and exists in no frozen gate script; (b) *silently scoped* — the pooling excludes T4, the one certified task where the ac-GRU wins decisively; (c) *narrowly supported* — two near-duplicate categorical cue-recall families on one DMC scene, one host, one scale, linear probes as the sole endpoint; and (d) *under-baselined* — the paper's own related-work section names the rivals (Gated DeltaNet, MesaNet, Mamba-family, RKN/ac-RKN) that were never run, the ac-GRU received none of the two generations of repair budget the candidate got, and the V18 review's behavior-level ask is unmet for the third consecutive cycle.

![Panel objections converted to V21 phases](figures/fig_v21_map.svg)

## 2. The V20 issues, itemized

**I1 — Epistemic labeling of the headline.** Claim 6 presupposed the opposite sign; the inversion fired the registered moot clause, and its pooled statistic is descriptive. A preregistration-discipline paper cannot present a non-registered p-value as confirmatory without self-contradiction. (Also: the pooled contrast script must be frozen; today it lives in an analysis snippet.)

**I2 — The T4 exclusion.** The rfix family's t4 ridge-R² collapse (−3.2 to −4.2 vs ac-GRU's −0.37) is unresolved — information loss or readout fragility is unknown — and the headline is computed without the one task whose sign reverses. Until a scale-robust continuous probe family adjudicates this, any "certified-memory tasks" generality claim is unearned.

**I3 — Probe-only endpoints, and the law they contaminate.** Every V19/V20 endpoint is a linear probe on `prior_read`. V20's own Insight V20.5 shows linear probes absorb trust rescaling — which means both the headline win and the adaptation negative are conditioned on a readout family the paper itself proved partially blind. The "four-fold trust-timescale law" is thereby part tautology: *calibration pays only where a consumer exists* was guaranteed by an evaluation family that consumes none. The V18 reviewer's behavior-level ask (registered as V19 Tier-2 gate 6, CEM/MPC — never executed) is precisely the missing consumer.

**I4 — Baseline asymmetry and missing rivals.** The LKC got two generations of diagnosis-driven repair; the ac-GRU is one untuned recipe (absolute scores 0.37–0.50 against a certified ~1.0 sighted ceiling — *both* arms recover under half the available information). No modern gated linear-recurrence cell, no RKN-lineage baseline, no long-context transformer control, and the ac-SSM was dropped after V19.

**I5 — Zero external validity.** The MIKASA-Robo arm registered in V19 §4.4 never ran; s\* is a single-host observation; no delay-scaling curves; one scene, one encoder scale, one data scale.

**I6 — Over-broad TTA claim.** C5e ("derived gain dominates fixed η") was measured in a φ-only adaptation family, not AdaJEPA's actual recipe, on streams the program itself certified post-hoc as requiring no adaptation. It must be scoped to "among adaptors, on no-demand streams — and zero updates beat both," and the "every AdaJEPA-style system should swap" prescription deleted until drift-demand-certified streams exist.

**I7 — Six-contribution sprawl and internal framing.** "First confirmed win in twenty generations" has no external meaning and invites the benchmark-co-evolution reading; the VisReg host study belongs in an appendix; the confirmatory layer must contain only frozen-gate survivors (C5e; claim 3's calibration transfer).

## 3. The V21 kernel

![V21 kernel: give the belief a consumer](figures/fig_v21_kernel.svg)

The program's kernel says memory is a belief filter over exogenous latents. V19 certified the *tasks* (memory demand exists), V20 certified the *filter* (structure validated, trust must be slow, calibration restorable on frozen weights). What was never certified is the *point of having a belief*: every endpoint so far asks "can a linear probe read ξ out of the belief?" — a question that consumes the mean and discards the variance, and that V20 proved is partially blind by construction. V21 closes the loop:

```
Level 0 (training)      VICReg host                     — frozen, validated
Level 1 (per-frame)     LKC, slow fixed trust           — frozen, validated (the W3 checkpoints are reused as-is)
Level 2 (deployment)    OFF                             — falsified; excluded by registration
Level 3 (NEW)           the consumer:  a_t* = argmax E[ Σ r̂(z̃, m) | m_t, σ_t ]
                        belief-conditioned MPC on reward-bearing certified tasks
```

Reward-bearing task variants make the memory demand *behavioral*: T1-act (reward for reaching the marker the vanished cue indicated) and T4-act (reward for tracking the occluded target through the gap). The certificate becomes a **return-floor certificate** — an integrator-features policy earns ≈ floor return, an oracle-ξ policy defines the ceiling, and the gap between them is the certified memory demand *in reward units*. The reward head r̂(z̃, m) is trained to need the belief; the planner comes in two variants differing in exactly one thing — mean-only (consumes m) vs σ-aware (expected reward under the Gaussian belief, analytic for the quadratic proximity rewards) — so **the value of uncertainty, and hence of calibration, is finally priced in return**: the registered redemption test for V20's claim-3/5 split is `return(σ-aware, calibrated trust) > return(σ-aware, detuned trust)` on segments where belief variance is decision-relevant (the T4 gap), while probes cannot distinguish the two. Either outcome closes the question: the trust-timescale law becomes either a law about world models or a documented artifact of probe evaluation.

Everything else in V21 is epistemic repair in the panel's leverage order — no new mechanism anywhere.

## 4. Design

### X0 — free repairs and baseline parity (≈ 2–3 GPU-days, mostly re-analysis)
1. **Tier-0 sensitivity analysis** (0 GPU-days): recompute every W3 contrast excluding the 13/90 convergence-failed cells and, separately, the task×seed failure clusters; publish per-task seed-level CIs alongside the crossed bootstrap. 
2. **T4 probe-family repair** (~0.5 GPU-day, frozen checkpoints re-probed): standardized-target RidgeCV with a registered regularization path + a small MLP probe control; adjudicate information-loss vs readout-fragility; recompute all pooled contrasts with t4 included under the repaired family.
3. **Baseline parity** (~2 GPU-days): ac-GRU lr/width sweep at matched parameters; a **symmetric-repair control** — the slow-trust diagnosis applied to the baseline (chrono-init / frozen-gate-bias GRU); ac-SSM reinstated at n=10; one modern gated delta-rule cell (parameter-matched, action-conditioned) as the input-dependent-gain family member the related work names.

### X1 — the preregistered confirmation wave (~1–2 GPU-days)
One endpoint, registered before any X0 unblinding beyond the probe-family choice: **lkc_rfix > best-of-{tuned ac-GRU, slow-gate ac-GRU, ac-SSM, gated-delta cell}**, fresh seeds (10 new), pooling rule frozen in advance *including t4 under the repaired probe family*, computed by a frozen gate script, Holm-corrected with X2's claims. This converts the inversion from a moot-clause quantification into a confirmatory result — or falsifies it against a fair envelope, which is equally publishable given the chain of three prior directional replications.

### X2 — the consumer (~3–4 GPU-days + engineering)
Reward-bearing tasks (T1-act, T4-act) with return-floor certificates; reward head + CEM/MPC planner over the frozen carriers (open-loop latent rollouts, the V19 Tier-2 gate-6 design); arms = {lkc_rfix, best envelope member, none} × {mean-only, σ-aware} × {calibrated, detuned-r} — the full 2×2 that separates memory value, uncertainty value, and calibration value in return units; test-time carrier ablation as the causal check.

### X3 — portability (~5–7 GPU-days, highest risk, lowest leverage-per-day — last)
MIKASA-Robo subset (RememberColor, ShellGameTouch, InterceptGrab) through the certification pipeline — the certificates themselves are the demo; s\* ladder on the repo's `FrozenDINOEncoder` (second host, makes the threshold an instrument); delay-scaling curves (cue-to-decision delay swept via the existing task knobs) — the actual shape of a memory claim.

### Registered exclusions
Deployment adaptation (dead until drift-demand-certified streams exist — and building those is *not* in V21's critical path; C5e ships scoped as-is); the VisReg host line (parked; s\* instrument ready if target separation is ever built); learned timescales, hierarchies, per-frame trust, training-time NLL (frozen defaults, four falsifications deep).

## 5. Claims ladder

| # | Claim | Phase | Confirmed if | Falsified if |
|---|---|---|---|---|
| 1 | The inversion is robust to analysis choices | X0 | survives cell-exclusion sensitivity + repaired t4 pooling | headline was carried by unhealthy cells / probe fragility → report and stop |
| 2 | The inversion is confirmatory against a fair envelope | X1 | frozen single-endpoint gate, fresh seeds, Holm | fair tuning closes the gap → V20's result was baseline-effort asymmetry (publishable correction) |
| 3 | Memory demand certifies in reward units | X2 | integrator-policy return ≈ floor, oracle-ξ ≫ floor | reward tasks admit a memoryless shortcut → fix task before any claim 4–5 |
| 4 | The probe-space advantage transfers to control | X2 | rfix > envelope in certified return | probe advantage is readout-specific → the field's probe-based memory evaluations are unsafe (a finding) |
| 5 | Uncertainty and calibration have decision value | X2 | σ-aware > mean-only where variance is decision-relevant; calibrated > detuned under σ-aware while probes can't tell | the trust-timescale law is evaluation-family-conditioned → registered as such, V20.5 closed |
| 6 | The result and the instruments are portable | X3 | certificates + inversion direction on ≥ 1 external family; s\* reproduces on a second host; advantage grows with delay | any leg fails → scope the paper accordingly (each leg reportable alone) |

## 6. Phasing, cost, and the paper strategy

X0 → X1 → X2 → X3, ≈ 11–16 GPU-days total on the 3-GPU budget, W3 checkpoints reused throughout (X0–X1 retrain only baselines; X2 trains reward heads only).

**Paper strategy (area-chair recommendation, adopted):** split with a re-merge option. **Paper A** after X0–X1 (~1 week): methodology-led — *certified memory demand: proving your benchmark requires memory, your encoder kept the evidence, and your drift required adaptation* — with the properly-registered inversion as the case study; panel-estimated 7–8. **Paper B** after X2–X3: the architecture claim, carried by return-level endpoints, the fair envelope, the external arm, and delay scaling. If X2 lands positive early, A and B re-merge into one strong submission; the reverse operation does not exist, which is the argument for the split. Framing rules bound both papers: confirmatory layer = frozen-gate survivors only; the V20 inversion presented as a registered-moot outcome with its three-dataset directional replication chain (P3 +0.086 4/5 → W1 6/6 → W3 19/20); C5e scoped to "among adaptors, on no-demand streams"; internal-program language ("first win in twenty generations") deleted; VisReg host study to the appendix.

## 7. Key sources

Everything inherits `docs/V20_PROPOSAL.md` §7 and `docs/V19_PROPOSAL.md` §13. New for V21: MIKASA-Robo arXiv:2502.10550 · POPGym arXiv:2303.01859 · RKN (Becker et al., ICML 2019) and ac-RKN (CoRL 2020) — the baseline lineage claim 2's envelope must include by name · Gated DeltaNet arXiv:2412.06464, MesaNet arXiv:2506.05233 (the input-dependent-gain family) · chrono initialization (Tallec & Ollivier, 2018) for the symmetric-repair control · CEM/MPC world-model planning conventions per the repo's V19 Tier-2 gate-6 registration.

---

**Awaiting review.** On approval, implementation begins at X0; the panel artifacts (four reviews + area-chair synthesis) are preserved in the session workflow transcript for audit.
