# Certify Before You Claim: Demand Certificates for Memory in Latent World Models

## Abstract

Claims about memory in latent world models rest on three premises that are almost never checked: that the benchmark actually requires memory, that the encoder retained the evidence to be remembered, and that the probe used to read the memory out reflects what a decision-maker could use. We introduce a certification protocol that tests each premise before any architecture claim is made: a **two-sided memory-demand certificate** (a sighted probe must read the cue while an initial-observation integrator and a decision-time probe must not), a **salience-threshold instrument** $s^*$ that measures per (encoder, scene) pair which cues survive representation learning, and a **return-floor certificate** that restates memory demand in reward units. We validate the protocol on a preregistered case study — a derived Kalman-style carrier versus a sixteen-config fair envelope of learned recurrence, confirmed at pooled d = +0.996 (p = 5.0e-05) on fresh seeds under a frozen gate, transferring to executed control at 0.769 vs 0.508 success under belief-conditioned goal selection with oracle execution — and then take the instruments to a world we did not build. On first contact, the certificate catches a memoryless shortcut in an external memory benchmark: in MIKASA-Robo RememberColor, the cue is visible at $t{=}0$, so an integrator that stores its first observation decodes the answer at 0.89 (chance 0.111) — the task demands initial-state storage, not online filtering. The protocol also audits itself: a second-scene deconfound overturned our own single-scene attribution of encoder blindness, and probe-level rankings twice failed to predict control-level rankings. We release the certificates, the instrument, and a fully registered record including every negative result.

## 1. Introduction

A memory study in latent world models typically proceeds: pick a partially observable benchmark, train an architecture with a recurrent state, probe or evaluate for retained information, and report the difference against a baseline. Every step of this recipe presupposes something the paper does not check. The benchmark is presumed to require memory — but if the identifying cue is still visible at decision time, or derivable from the initial observation plus the action history, a memoryless policy solves it. The encoder is presumed to have kept the evidence — but self-supervised objectives can delete low-salience factors before any memory module sees them. And the probe is presumed to speak for behavior — but a linear readout on a belief state answers a different question than a decision-maker consuming that belief.

![The two-sided memory-demand certificate on an episode timeline, annotated with our first external audit (§5.5): the sighted and leakage probes pass, but the integrator floor decodes $\xi$ far above chance because the cue is visible at $t{=}0$ — the task demands storage, not filtering.](figures/fig_a_protocol.png)

This paper makes the checking itself the contribution. We develop and validate three instruments — a **two-sided memory-demand certificate** (Figure 1), a **salience-threshold instrument** $s^*$ measured per (encoder, scene) pair, and a **return-floor certificate** that restates demand in reward units (§4) — and we bind them to every stage of a world-model memory pipeline (Figure 2). We validate the protocol on a preregistered architecture comparison (§5.2–5.4) whose result we state carefully: a derived filter with a slow fixed trust beats the best of sixteen fairly tuned learned-recurrence configurations on certified probe endpoints (pooled d = +0.996, frozen gate, fresh seeds), and the advantage transfers to executed control under belief-conditioned goal selection with oracle dynamics (0.769 vs 0.508 vs ceiling 0.917). We then take the instruments outside the worlds we built (§5.5–5.6): they audited an external benchmark on first contact and caught a memoryless shortcut its own framing misses, and they audited *us*, twice — a second-scene control withdrew our encoder-blindness mechanism claim, and probe rankings failed to predict control rankings within the learned-recurrence family.

We consider the negative and self-corrective results part of the contribution, and we report them in the main text: the return-level evaluation could not use the latent world model because the host cannot roll forward one useful step (a *rollout-competence* prerequisite we register as a third demand certificate, §5.3); the filter's delay advantage shrinks under extrapolation and is not rescued by re-deriving its spectrum for the horizon; per-decision uncertainty consumption was worth approximately nothing on our tasks.

## 2. Preliminaries

An episode is a frame sequence $o_{0:L-1}$ with actions $a_{0:L-2}$ and an exogenous latent $\xi$ (categorical unless noted), rendered with a cue window $[t_{on}, t_{off}]$ inside which $\xi$ is visible. A bank is $E$ episodes from one (task, seed) cell. All probes are the registered family — logistic for categorical $\xi$, standardized-target RidgeCV over a registered regularization path for continuous $\xi$; §5.2 explains, with a cautionary tale, why the probe family must be registered.

**Testbed.** One DeepMind-Control reacher scene \citep{r_dmc} with exogenous cue overlays composited onto the pixels, three certified task families deep: **t1** (a transient cue flashes at one of four markers during a random early window; categorical $\xi$, chance 0.25), **t3** (a drifting distractor whose class must be recalled; three classes), and **t4** (an occluded target tracked through a gap; continuous $\xi$). Episodes are 64 frames; the cue is over by frame 22 and the registered probes read the belief in the post-cue window and at the final frame, so the bridged delay is roughly 50 steps. Hosts are VICReg-style pixel encoders \citep{r_vicreg} trained per task; carriers consume frozen encodings. A second scene (point-mass, same overlay family) and an external benchmark (MIKASA-Robo RememberColor9, frozen DINOv2 host \citep{r_dinov2}) enter in §5.5–5.6.

**Carriers.** The candidate is a latent Kalman cell with a fixed log-spaced decay spectrum, an exact eigenvalue-1 hold channel, and slow fixed trust ("the filter," rfix). The learned envelope contains GRU variants (including a chrono-initialized slow-gate control), an action-conditioned SSM, and a parameter-matched action-conditioned gated delta-rule cell. We claim no novelty for either family; the comparison exists to exercise the certificates, and we state its scale honestly: about 1,200 training episodes, one scene family, both arms recovering under half the sighted ceiling.

## 3. Related work

**Memory benchmarks.** POPGym \citep{r_popgym}, Memory Gym \citep{r_memgym}, and MIKASA-Robo \citep{r_mikasa} supply memory-labeled task suites; MIKASA-Robo's demand notion is sliding-window observability. Our certificate is stricter and two-sided: it grants the adversary the *entire* action history and the initial observation (the strongest memoryless policy that is causally legal), and separately verifies the evidence survived encoding. The MDP/POMDP twin design of POPGym Arcade \citep{r_arcade} is the closest antecedent of the demand side; the sighted/leakage sides and the per-encoder salience instrument are, to our knowledge, new. §5.5 shows the difference matters: a task that is memory-demanding under sliding-window observability certifies as storage-demanding, not filtering-demanding, under ours.

**World models with memory.** Recurrent state-space models and their successors \citep{r_planet,r_dreamer,r_dreamerv3,r_tdmpc2,r_r2i} report return-level results with memory in imagination; pixel JEPA world models \citep{r_lewm,r_dinowm} supply the host family we study. We do not compete with these systems; we supply the audit layer they skip. Our return-level endpoint (§5.3) deliberately isolates the memory factor by planning with oracle dynamics after the host failed a rollout-competence check — which is itself a result those systems' evaluation protocols would not have surfaced, because multi-step training objectives are baked in rather than certified.

**Linear recurrence and derived filters.** The envelope's delta-rule cell follows the input-dependent-gain family \citep{r_gdn,r_mesanet}; the chrono control follows \citet{r_tallec}; the filter is HiPPO-adjacent in construction \citep{r_hippo} and RKN-lineage \citep{r_rkn}.

**Probing and its limits.** Work on probing representations has long warned that decodability is not use \citep{r_belinkov,r_ravich}. We give this warning a quantitative instance in world models: two independent cases where probe-level and control-level measurements dissociate — the probe-selected envelope winner loses in control, and calibration damage that probes absorb costs a quarter of executed success — both in §5.4, and both found by an adversarial verification pass on our own program.

## 4. Methodology

![The certification protocol bound to a world-model memory pipeline. Gray: the standard stack every memory study already has. Blue: the two readout levels. Green: the certificates — each gates the stage it points to, and each has caught at least one wrong claim (Appendix A).](figures/fig_a_arch.png)

### 4.1 Two-sided memory demand

A bank is **memory-demanding at probe level** iff:

1. *Sighted*: a probe on encoder features of eight frames spanning the cue window decodes $\xi$ at $\geq 0.75$. This certifies the evidence exists *after encoding* — the step almost every memory paper skips, and the step that failed silently in our own program's history (§5.6).
2. *Integrator floor*: a probe on $[\mathrm{enc}(o_0);\, a_{0:L-2}]$ scores within 0.05 of chance. This adversary retains its first observation forever and integrates all controls — the strongest memoryless reference that requires no memory of *observations after* $t{=}0$.
3. *No leakage*: a probe on decision-window frames scores within 0.05 of chance — $\xi$ must not be re-derivable at decision time.

One side without the other licenses nothing: a failed sighted probe means the encoder deleted the evidence; a passing floor means the task never needed memory. The certificate is per-(task, encoder, seed) and fails closed: our pipeline refuses to score memory endpoints on banks that have not passed it.

### 4.2 The salience instrument $s^*$

Fix a task family with a monotone salience knob (cue size, duration, border) and a probe gate. $s^*(\text{encoder}, \text{scene})$ is the lowest rung whose sighted certificate passes (majority over bank seeds). Two properties make it an instrument rather than a score: it is *comparable across encoders on a fixed scene* (same banks, same probe), and *diagnostic before training a single memory module* — if the cue you intend to test memory on sits below $s^*$, the study is unrunnable regardless of architecture.

### 4.3 Memory demand in reward units, and rollout competence

Attach a reward to the recalled factor (reach the marker the vanished cue indicated). The **return-floor certificate** requires: an oracle-$\xi$ policy achieves ceiling return; an integrator-features policy achieves floor return; and the gap clears a registered margin (0.67 $\gg$ 0.3 here). The gap *is* the memory demand, in the units decisions are made in. Any behavior-level evaluation that plans through a learned model additionally presupposes the model can roll forward; we register **rollout competence** — decoded task state must survive $k$ predicted steps — as the corresponding prerequisite certificate, having failed it ourselves (§5.3).

## 5. Experimental results

### 5.1 Overview: seven registered questions, seven adjudicated answers

Every endpoint below was registered — endpoint, bar, and falsified-clause — before execution; the released record carries the timestamps. Table 1 is the map of this section.

| § | registered question | outcome |
|---|---|---|
| 5.2 | is the probe-level advantage robust to analysis choices? | confirmed (strengthens under exclusions) |
| 5.2 | does it survive a fair envelope, fresh seeds, frozen gate? | **confirmed**, d = +0.996 |
| 5.3 | does memory demand certify in reward units? | confirmed (gap 0.67) |
| 5.3 | does the advantage transfer to executed control? | confirmed, n = 3, direction-only |
| 5.4 | do uncertainty and calibration have decision value? | split: encode-time yes, per-decision no |
| 5.5 | do the certificates work on an external benchmark? | yes — informative failure (shortcut caught) |
| 5.6 | do the instruments port across encoders and scenes? | ported; our own mechanism claim withdrawn |

Table: The registered questions of §5 and their adjudicated outcomes; every bar and falsified-clause was frozen before execution.

### 5.2 Probe-level: repairs, then a frozen confirmation gate

**Repairs first (frozen checkpoints, no new data).** Excluding every (task, seed) cell that failed registered training-health gates *strengthens* the contrast: full grid +0.1225 (p = 1.0e-05) → healthy-only +0.1308 (p = 9.5e-07, 13/14). And the hardest task's apparent pathology (ridge $R^2$ of -3.15 for the filter) was a readout artifact: under a scale-robust registered family, the filter *leads* on that task too (+0.153 vs +0.094); with the task rejoined, the pooled contrast is d = +1.84. The lesson we register for the field: *the probe family is part of the claim* — an unregularized readout manufactured a −3 to −4 $R^2$ "collapse" that survived two program generations.

**The envelope.** Sixteen parameter-matched configurations — a GRU lr×width sweep, a chrono-initialized slow-gate GRU (the filter's own "slow trust" diagnosis applied symmetrically to the baseline), an action-conditioned SSM, and an action-conditioned gated delta-rule cell — selected on dev banks by a rule registered before any sweep result existed. Winner: the delta cell (0.635 pooled dev vs 0.588 for the best GRU; the chrono repair did **not** rescue the GRU, 0.551).

**The gate.** Endpoint: pooled standardized paired d of (filter − envelope-winner) over three certified tasks under the repaired probe family, ten fresh seeds; confirmation iff bootstrap $p_{pos} < 0.05$ and $\geq 2/3$ tasks positive; the falsified clause ("fair tuning closes the gap") registered alongside. The gate script refuses artifacts predating the registration. **Result: confirmed** (Table 2) — pooled d = +0.996, CI95 [+0.637, +1.899], $p_{pos}$ = 5.0e-05.

| task | mean diff (filter − delta cell) | seed wins | paired d |
|---|---|---|---|
| t1 | +0.1247 | 10/10 | +1.98 |
| t3 | +0.0383 | 7/10 | +0.52 |
| t4 | +0.0623 | 7/10 | +0.49 |

Table: The frozen confirmation gate, per task (ten fresh seeds, repaired probe family).

**Disclosure, unprompted.** A post-hoc audit found every envelope-winner training cell fails our registered representation-health gates (effective rank 3.0–10.7 against a minimum of 16) while the filter passes 27/30. Either the rank gate does not transfer to matrix-state cells — plausible: the delta cell's readout is a projection of a rank-limited state — or the confirmed rival is a degenerate trainee. We report the confirmation *with* this asymmetry attached and register a delta-cell-appropriate health criterion as a precondition for any future architecture-headline use of this result.

### 5.3 Control-level: the advantage transfers — stated exactly

The reward-bearing variant (T1-act: reach the marker the vanished cue indicated) passes the return-floor certificate: ceiling 0.917, floor 0.244, gap 0.67. Evaluation is **belief-conditioned goal selection under oracle execution**: a frozen linear selector reads each carrier's belief at plan time, and an identical CEM planner executes under oracle physics. We name the substrate honestly because the registered latent-planning design died first: the host world model transports pose information for less than one predicted step (decode error 0.104 on real frames → 0.469 after one predicted step → no information by four). We failed our own §4.3 prerequisite and isolate the memory factor with oracle dynamics — the only thing differing between arms is the belief. Table 3 gives every arm under the identical planner.

| arm (identical planner) | executed success | role |
|---|---|---|
| oracle-$\xi$ | 0.917 | ceiling |
| **filter (rfix)** | **0.769** | candidate |
| filter, trust detuned $\times$16 | 0.500 | calibration ablation |
| GRU | 0.508 | registered comparison |
| delta cell (exploratory, post-registration) | 0.339 | envelope winner by probes |
| no-carrier selector | 0.256 | = chance |
| belief ablated (uniform weights) | 0.064 | causal check |
| integrator floor | 0.244 | certificate floor |

Table: Executed success under the identical oracle-execution planner; only the belief source differs between arms (mean of 3 checkpoint seeds, 120 episodes each).

The registered gate (filter > GRU, paired over checkpoint seeds) passes 3/3; we state its statistical weight honestly — n = 3 checkpoint seeds, direction and margin, no p-value — and note the margin (+0.26 success) is the probe gap passed through the argmax, not a new information source. The causal checks hold in both directions: removing the carrier collapses to chance, ablating the belief collapses below it. (One further disclosure: a full-execution re-roll reads systematically higher than the planning-environment estimate by three to four points; reported, not corrected.)

### 5.4 Probes do not rank what control ranks

Two exploratory results, reported because they audit the field's default evaluation. (i) The envelope winner *by probe selection* loses *in control*: the delta cell's executed success is 0.339 — below the GRU (0.508) it beat on dev probes — with one seed's plan-time selector at chance (0.225). (ii) Miscalibrating the filter's trust costs -0.27 success (Table 3) and is tracked by belief informativeness at plan time (selector accuracy 0.825 → 0.536, persisted artifact), consistent with full mediation, while per-decision uncertainty consumption — hedging over goals by belief weights — is worth +0.003 calibrated (+0.014 detuned). Calibration is priced at *encode time*, not read time, on these tasks; and probe-level rankings within an architecture family did not predict control-level rankings. Neither observation survives our own confirmatory bar; both are exactly the kind of observation the field's probe-only evaluations cannot make.

### 5.5 First contact with an external benchmark: the certificate catches a shortcut

We ran MIKASA-Robo RememberColor9 (nine colors; cue cube shown, hidden, then nine candidates presented) through §4.1 with a frozen DINOv2 host, three bank seeds, xi-independent random actions (2,304 episodes total; Table 4).

| certificate side | result (3 banks) | verdict |
|---|---|---|
| sighted (cue window) | 1.000/1.000/1.000 | PASS — cue perfectly encoded |
| no-leakage (decision window) | 0.086–0.125 (chance 0.111) | PASS — decision phase clean |
| integrator floor | **0.859–0.930** | **FAIL — memoryless shortcut** |

Table: The two-sided certificate on MIKASA-Robo RememberColor9 (frozen DINOv2 host): the evidence is encoded and the decision phase is clean, but a first-observation integrator decodes the answer — a storage task wearing a memory label.

The floor fails because the cue cube is visible at $t{=}0$ (Figure 1). Under MIKASA's sliding-window demand notion the task requires memory; under ours it requires *initial-observation storage*, not online filtering — an agent architecture with any $o_0$-conditioning (a goal encoder, a first-frame feature cache) has the information to bypass the intended difficulty. We report this not as a defect of MIKASA-Robo but as the demonstration the protocol exists for: **the two demand types are distinguishable only by certificate, and a benchmark's "memory" label can conflate them — as it does here.** (The task family is easily repaired — delay the cue onset — and the certificate verifies the repair.) An exploratory transfer of the §5.2 comparison to this family (carriers trained on frozen DINO features with a documented transfer recipe, n = 5, direction-only): filter 0.845 (per-seed sd 0.004) vs delta cell 0.600 (sd 0.23) — the direction replicates externally, and so does the delta cell's seed instability.

### 5.6 The instruments abroad: a self-correction and a scoped negative

**Encoder deletion is scene-dependent — our own claim withdrawn.** Our program's prior generation reported "encoder blindness is acquired, not architectural": the task-trained VICReg encoder fails the sighted certificate at low salience ($s^* = $ t1s2; scores 0.297/0.570/0.746 at the rung below) where frozen DINOv2 passes everything. The claim was confounded — architecture, pretraining data, and objective all differed — so we registered a deconfound with the interpretation table frozen in advance: train the *identical* VICReg recipe on a second scene's version of the same salience ladder. **The correction branch fired** (Figure 3a): on the second scene, the task-trained recipe *passes* the rung it fails on the original scene (0.871 / 0.789, floors at chance, health gates passing). The deletion is **scene-dependent** — consistent with a registered variance-budget-competition hypothesis (§7), not yet a mechanism — and our attribution is withdrawn as registered. Meanwhile the DINOv2 ladder passes every rung on both scenes at $\geq 0.988$, including a two-pixel-radius cue lasting two to three frames, so $s^*(\mathrm{DINOv2})$ is reported as an *upper bound* ($\leq$ t1s0c), not a threshold: the instrument bottoms out at the scene's render floor before the frozen backbone does. What survives is the instrument thesis in its strongest form: **whether an encoder deletes your cue is a property of the (encoder, scene) pair that cannot be predicted from the recipe — it must be certified per pair,** and the certificate that told us we were wrong is the same certificate we are proposing.

![Left: the $s^*$ instrument across two hosts and two scenes (mean over bank seeds, whiskers min–max) — the task-trained VICReg encoder fails the s1 rung on reacher yet passes it on point-mass, withdrawing our "acquired, not architectural" attribution, while frozen DINOv2 saturates both scenes. Right (exploratory): registered-probe accuracy vs episode length (frozen checkpoints, fresh banks; whiskers ± sd) — the filter leads at every delay, all carriers decay toward chance beyond the training length against the registered grows-with-delay expectation, and the registered repair (spectrum re-derived per horizon, dashed) recovers nothing: the decay is readout-limited, not retention-limited.](figures/fig_a_results.png)

**Delay scaling, and a negative on the obvious repair** (Figure 3b). The registered expectation — that the filter's advantage *grows* with delay — was **not met**: the advantage is positive at every tested cue-to-decision delay but narrows under extrapolation. We report the delay curves as exploratory scoping: 0.451 / 0.380 / 0.310 (filter) vs 0.317 / 0.307 / 0.263 (GRU) vs 0.328 / 0.323 / 0.272 (delta cell) at L = 64/96/128 (training length 64, chance 0.25). The natural repair — the filter's spectrum is a design knob, so re-derive its half-lives for the evaluated horizon on frozen weights — was registered with a confirmation bar and **failed it** (+0.002 at L = 128; the L = 64 no-op sanity check returned +0.000 exactly). The decay is not retention-limited (the cue rides an exact eigenvalue-1 hold channel); it is *readout*-limited — the learned weights are fit to length-64 statistics. We report the memory claim scoped to delays near the training regime, with the mechanism for why extrapolation fails attached.

## 6. Conclusion

Before a memory claim, three questions: does the task demand memory (both sides certified)? did the encoder keep the evidence ($s^*$ for your pair)? does your readout speak for decisions (return-floor, not probes)? Our record shows each question, unasked, has produced a wrong published-grade claim — including two of our own, caught by our own instruments and corrected under frozen registrations in this paper. The certificates are cheap, the instrument is one ladder, and the alternative is a literature of memory results that may be storage results, encoder results, or readout results in disguise.

## 7. Limitations

All confirmatory results live on one scene family plus one external task; the second scene enters only through the $s^*$ grid. The control-level endpoint uses oracle dynamics by measured necessity, and its registered gate is n = 3, direction-only. The envelope's winner fails our representation-health gates (disclosed in §5.2); a delta-cell-appropriate criterion is registered before any architecture headline. $s^*(\mathrm{DINOv2})$ is censored at the render floor. The scene-dependence of encoder deletion is an observation with a registered hypothesis (variance-budget competition), not a mechanism. Forward registrations, frozen in the released record: the variance-relevant uncertainty test on a rollout-competent host; a scene-complexity sweep for the deletion hypothesis; Holm-corrected joint families for any future confirmatory wave.

APPENDIXMARKER

## What the certificates caught: a summary

Across three program generations and one external benchmark, the protocol's catches — each of which would have shipped as a wrong or unscoped claim without it:

| # | catch | certificate | consequence |
|---|---|---|---|
| 1 | tasks that never required memory (prior generation; released record) | integrator floor | benchmark rebuilt |
| 2 | encoders that deleted the cue before memory saw it | sighted | host preflight now mandatory |
| 3 | a −3 to −4 $R^2$ "pathology" that was the readout, not the model | registered probe family | hardest task rejoined the pool, sign unchanged |
| 4 | a world model that cannot roll forward one step | rollout competence (new) | latent-planning claims blocked, substrate renamed |
| 5 | probe rankings that do not predict control rankings (×2) | return-floor | probe-only memory evaluations flagged |
| 6 | an external "memory" task solvable by storing frame 0 | integrator floor | demand-type taxonomy (storage vs filtering) |
| 7 | our own encoder-blindness mechanism claim | second-scene $s^*$ | claim withdrawn, rescoped to per-pair certification |

Table: Every catch the protocol made across three program generations and one external benchmark.

## External-audit details

RememberColor9 phase structure: the cued cube is visible at the scene center for steps $[0, 5)$, all cubes are hidden for $[5, 10)$, and all nine candidates are visible at shuffled positions from step 10 with the cued color unmarked. Banks: three seeds × (512 train + 256 eval) episodes at 60 steps, 64×64 frames, xi-independent uniform random actions. The dependency stack pins a `numpy` version without Python 3.12 wheels; the release scripts the dedicated-environment workaround that unblocked the arm after three prior registrations failed to run it. The exploratory carrier transfer trains each carrier on frozen DINOv2 features with a next-feature prediction objective (the feature-space form of the program's residual-read objective) and probes the belief with the registered family (delay-window mean plus decision-onset read).

## Reproducibility statement

Every number in this paper is injected from a hash-manifested artifact by the release renderer; the manuscript manifest binds artifact SHA-256s. The full program record — registrations (timestamped before execution), amendment trails, adversarial review, negative results, and the same-day corrections — ships as the supplement. All experiments ran on three consumer-grade GPUs.
