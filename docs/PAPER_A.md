# Certify Before You Claim: Demand Certificates for Memory in Latent World Models

## Abstract

Claims about memory in latent world models rest on three premises that are almost never checked: that the benchmark actually requires memory, that the encoder retained the evidence to be remembered, and that the probe used to read the memory out reflects what a decision-maker could use. We introduce a certification protocol that tests each premise before any architecture claim is made: a **two-sided memory-demand certificate** (a sighted probe must read the cue while an initial-observation integrator and a decision-time probe must not), a **salience-threshold instrument** $s^*$ that measures per (encoder, scene) pair which cues survive representation learning, and a **return-floor certificate** that restates memory demand in reward units. We validate the protocol on a preregistered case study — a derived Kalman-style carrier versus a sixteen-config fair envelope of learned recurrence, confirmed at pooled d = +0.996 (p = 5.0e-05) on fresh seeds under a frozen gate, transferring to executed control at 0.769 vs 0.508 success under belief-conditioned goal selection with oracle execution — and then take the instruments to worlds we did not build. On first contact, the certificate catches a memoryless shortcut in an external memory benchmark: in MIKASA-Robo RememberColor, the cue is visible at $t{=}0$, so an integrator that stores its first observation decodes the answer at 0.89 (chance 0.111) — the task demands initial-state storage, not online filtering. The protocol also audits itself: a second-scene deconfound overturned our own single-scene attribution of encoder blindness, and probe-level rankings twice failed to predict control-level rankings. We release the certificates, the instrument, and a fully registered record including every negative result.

## 1. Introduction

A memory study in latent world models typically proceeds: pick a partially observable benchmark, train an architecture with a recurrent state, probe or evaluate for retained information, and report the difference against a baseline. Every step of this recipe presupposes something the paper does not check. The benchmark is presumed to require memory — but if the identifying cue is still visible at decision time, or derivable from the initial observation plus the action history, a memoryless policy solves it. The encoder is presumed to have kept the evidence — but self-supervised objectives can delete low-salience factors before any memory module sees them. And the probe is presumed to speak for behavior — but a linear readout on a belief state answers a different question than a decision-maker consuming that belief.

This paper makes the checking itself the contribution. We develop and validate three instruments:

**The two-sided memory-demand certificate** (§3.1). A task instance is *memory-demanding* only if (i) a *sighted* probe on encoder features during the cue window reads the latent factor $\xi$ (the evidence exists and survived encoding), (ii) an *integrator floor* — a probe on the initial observation plus the executed action stream — scores at chance ($\xi$ is not derivable without observing the cue), and (iii) a decision-window probe scores at chance ($\xi$ has not leaked into later frames). One side without the other licenses nothing: a failed sighted probe means the encoder deleted the evidence; a passing floor means the task never needed memory.

**The salience-threshold instrument $s^*$** (§3.2). For a graded ladder of cue saliences, $s^*$ is the lowest salience at which the sighted certificate passes. It is a property of an (encoder, scene) pair, not of a task: we show the same ladder yields $s^*{=}$ t1s2 for a task-trained VICReg encoder on one scene, an upper bound of t1s0c for frozen DINOv2 on the same scene, and — the honest surprise — a *pass* for the identical VICReg recipe on a second scene, overturning our own registered mechanism claim (§5.2).

**The return-floor certificate** (§3.3). Probe decodability is not decision value. Rebuilding the demand certificate in reward units — an oracle-informed policy defines the ceiling, an integrator-informed policy the floor — lets memory claims be made at the level a reviewer of world models should demand: executed return.

We validate the protocol on a preregistered architecture comparison (§4) whose result we state carefully: a derived filter with a slow fixed trust beats the best of sixteen fairly tuned learned-recurrence configurations on certified probe endpoints (pooled d = +0.996, frozen gate, fresh seeds), and the advantage transfers to executed control under belief-conditioned goal selection with oracle dynamics (0.769 vs 0.508 vs ceiling 0.917). We then demonstrate that the instruments travel (§5): they audited an external benchmark on first contact and caught a memoryless shortcut its own framing misses, and they audited *us*, twice — a second-scene control withdrew our encoder-blindness mechanism claim, and probe rankings failed to predict control rankings within the learned-recurrence family.

We consider the negative and self-corrective results part of the contribution, and we report them in the main text: the return-level evaluation could not use the latent world model because the host cannot roll forward one useful step (a *rollout-competence* prerequisite we register as a third demand certificate, §6); the filter's delay advantage shrinks under extrapolation and is not rescued by re-deriving its spectrum for the horizon; per-decision uncertainty consumption was worth approximately nothing on our tasks.

## 2. Related work

**Memory benchmarks.** POPGym [Morad et al., 2023], Memory Gym [Pleines et al., 2023], and MIKASA-Robo [Cherepanov et al., 2025] supply memory-labeled task suites; MIKASA-Robo's demand notion is sliding-window observability. Our certificate is stricter and two-sided: it grants the adversary the *entire* action history and the initial observation (the strongest memoryless policy that is causally legal), and separately verifies the evidence survived encoding. POPGym Arcade's MDP/POMDP twin design is the closest antecedent of the demand side; the sighted/leakage sides and the per-encoder salience instrument are, to our knowledge, new. §5.1 shows the difference matters: a task that is memory-demanding under sliding-window observability certifies as storage-demanding, not filtering-demanding, under ours.

**World models with memory.** Recurrent state-space models and their successors [Hafner et al., 2019–2023; Hansen et al., 2024; Samsami et al., 2024] report return-level results with memory in imagination. We do not compete with these systems; we supply the audit layer they skip. Our return-level endpoint (§4.3) deliberately isolates the memory factor by planning with oracle dynamics after the host failed a rollout-competence check — which is itself a result those systems' evaluation protocols would not have surfaced, because multi-step training objectives are baked in rather than certified.

**Linear recurrence and derived filters.** The learned envelope in §4 includes a parameter-matched action-conditioned gated delta-rule cell [Yang et al., 2024; von Oswald et al., 2025 lineage], GRU variants including a chrono-initialized slow-gate control [Tallec & Ollivier, 2018], and an action-conditioned SSM. The derived carrier is a latent Kalman cell with a fixed log-spaced spectrum and an exact hold channel (HiPPO-adjacent construction; RKN lineage [Becker et al., 2019]). We claim no novelty for either family; the comparison exists to exercise the certificates, and we state its scale honestly: ~1,200 training episodes, one scene family, both arms recovering under half the sighted ceiling.

**Probing and its limits.** Work on probing representations has long warned that decodability is not use [Belinkov, 2022; Ravichander et al., 2021]. We give this warning a quantitative, preregistered instance in world models: two independent cases where probe-level and control-level rankings dissociate (§4.4, §5.3), one of which survived an adversarial verification pass on our own program.

## 3. The certification protocol

Notation: an episode is a frame sequence $o_{0:L-1}$ with actions $a_{0:L-2}$ and an exogenous latent $\xi$ (categorical here), rendered with a cue window $[t_{on}, t_{off}]$ inside which $\xi$ is visible. A bank is $E$ episodes from one (task, seed) cell. All probes are the registered family (logistic for categorical $\xi$; standardized-target RidgeCV for continuous — the repair in §4.1 explains why the family must be registered).

### 3.1 Two-sided memory demand

A bank is **memory-demanding at probe level** iff:

1. *Sighted*: a probe on encoder features of eight frames spanning the cue window decodes $\xi$ at $\geq 0.75$. This certifies the evidence exists *after encoding* — the step almost every memory paper skips, and the step that failed silently in our own program's history (§5.2).
2. *Integrator floor*: a probe on $[\mathrm{enc}(o_0);\, a_{0:L-2}]$ scores within 0.05 of chance. This adversary retains its first observation forever and integrates all controls — the strongest memoryless reference that requires no memory of *observations after* $t{=}0$.
3. *No leakage*: a probe on decision-window frames scores within 0.05 of chance — $\xi$ must not be re-derivable at decision time.

The certificate is per-(task, encoder, seed) and fails closed: our pipeline refuses to score memory endpoints on banks that have not passed it.

### 3.2 The salience instrument $s^*$

Fix a task family with a monotone salience knob (cue size, duration, border) and a probe gate. $s^*(\text{encoder}, \text{scene})$ is the lowest rung whose sighted certificate passes (majority over bank seeds). Two properties make it an instrument rather than a score: it is *comparable across encoders on a fixed scene* (same banks, same probe), and *diagnostic before training a single memory module* — if the cue you intend to test memory on sits below $s^*$, the study is unrunnable regardless of architecture.

### 3.3 Memory demand in reward units

Attach a reward to the recalled factor (reach the marker the vanished cue indicated). The **return-floor certificate** requires: an oracle-$\xi$ policy achieves ceiling return; an integrator-features policy achieves floor return; and the gap clears a registered margin (0.67 $\gg$ 0.3 here). The gap *is* the memory demand, in the units decisions are made in.

## 4. A certified case study, preregistered

The case study asks a deliberately narrow question: on banks that pass the §3 certificates, does a derived filter — a latent Kalman cell with fixed log-spaced decay spectrum, an exact hold channel, and slow fixed trust ("rfix") — retain the cue better than fairly tuned learned recurrence? The question matters here not for the architectures but because every step is gated by the certificates, and because its history is an argument for them: an earlier generation of this comparison was reported with the hardest task excluded and the baseline untuned, and an adversarial review of our own record (released with the paper) struck it. Everything below is the repaired, registered version.

### 4.1 Robustness repairs (before any new data)

Two objections were retired by reanalysis of frozen checkpoints. First, excluding every (task, seed) cell that failed registered training-health gates *strengthens* the probe-level contrast: full grid +0.1225 (p = 1.0e-05) → healthy-only +0.1308 (p = 9.5e-07, 13/14). Second, the excluded task's apparent pathology (ridge $R^2$ of -3.15 for the filter) was a readout artifact: under a scale-robust registered family (standardization + RidgeCV over a registered path), the filter *leads* on that task too (+0.153 vs +0.094), and the adjudication (fragility vs information loss) was made by a registered rule, not by choice. With the task rejoined, the pooled contrast is d = +1.84. The lesson we register for the field: *the probe family is part of the claim* — an unregularized readout manufactured a −3 to −4 $R^2$ "collapse" that survived two program generations.

### 4.2 The confirmation gate

The envelope: sixteen parameter-matched configurations — a GRU lr×width sweep, a chrono-initialized slow-gate GRU (the filter's own "slow trust" diagnosis applied symmetrically to the baseline), an action-conditioned SSM, and an action-conditioned gated delta-rule cell — selected on dev banks by a rule frozen *before* the sweep ran. Winner: the delta cell (0.635 pooled dev vs 0.588 for the best GRU; the chrono repair did **not** rescue the GRU, 0.551). The endpoint — pooled standardized paired d of (filter − envelope-winner) over three certified tasks under the repaired probe family, ten fresh seeds, confirmation iff bootstrap $p_{pos} < 0.05$ and $\geq 2/3$ tasks positive — was registered and timestamped before any sweep result existed, with the falsified clause ("fair tuning closes the gap") registered alongside. The gate script refuses artifacts predating the registration.

**Result: confirmed.** Pooled d = +0.996, CI95 [+0.637, +1.899], $p_{pos}$ = 5.0e-05; per-task d = +1.98 / +0.52 / +0.49, wins 10/10 / 7/10 / 7/10.

**Disclosure, unprompted.** A post-hoc audit found every envelope-winner training cell fails our registered representation-health gates (effective rank 3.0–10.7 against a minimum of 16) while the filter passes 27/30. Either the rank gate does not transfer to matrix-state cells — plausible: the delta cell's readout is a projection of a rank-limited state — or the confirmed rival is a degenerate trainee. We report the confirmation *with* this asymmetry attached and register a delta-cell-appropriate health criterion as a precondition for any future architecture-headline use of this result.

### 4.3 The advantage transfers to control — stated exactly

The reward-bearing variant (T1-act: reach the marker the vanished cue indicated; return-floor certificate: ceiling 0.917, floor 0.244, gap 0.67) was evaluated with **belief-conditioned goal selection under oracle execution**: a frozen linear selector reads each carrier's belief at plan time, and an identical CEM planner executes under oracle physics. We name the substrate honestly because the registered latent-planning design died first: the host world model transports pose information for less than one predicted step (decode error 0.104 on real frames → 0.469 after one predicted step → no information by four). We failed our own prerequisite, register **rollout competence** as a third demand certificate, and isolate the memory factor with oracle dynamics — the only thing differing between arms is the belief.

Under the identical planner: filter 0.769, GRU 0.508, no-carrier 0.256 (= chance), belief-ablated 0.064 (collapse below chance — the causal check in both directions). The registered gate (filter > GRU, paired over checkpoint seeds) passes 3/3; we state its statistical weight honestly — n = 3 checkpoint seeds, direction and margin, no p-value — and note the margin (+0.26 success) is the probe gap passed through the argmax, not a new information source.

### 4.4 Probes do not rank what control ranks

Two exploratory results, reported because they audit the field's default evaluation. (i) The envelope winner *by probe selection* loses *in control*: the delta cell's selector scores 0.339 — below the GRU it beat on dev probes — with one seed's selector at chance. (ii) Miscalibrating the filter's trust ($r \times 16$) costs -0.27 success and is fully mediated by belief informativeness at plan time (selector accuracy 0.825 → 0.536, persisted artifact), while per-decision uncertainty consumption — hedging over goals by belief weights — is worth +0.008. Calibration is priced at *encode time*, not read time, on these tasks; and probe-level rankings within an architecture family did not predict control-level rankings. Neither observation survives our own confirmatory bar; both are exactly the kind of observation the field's probe-only evaluations cannot make.

## 5. The instruments abroad

### 5.1 First contact with an external benchmark: the certificate catches a shortcut

We ran MIKASA-Robo RememberColor9 (nine colors; cue cube shown, hidden, then nine candidates presented) through §3.1 with a frozen DINOv2 host, three bank seeds, xi-independent random actions (2,304 episodes total). Sighted: 1.000/1.000/1.000 — the cue is perfectly encoded. Leakage: 0.086–0.125 (chance 0.111) — the decision phase is clean. **Integrator floor: 0.859–0.930** — a memoryless adversary that stores its first observation decodes the answer at ~0.9, because the cue cube is visible at $t{=}0$.

Under MIKASA's sliding-window demand notion the task requires memory; under ours it requires *initial-observation storage*, not online filtering — an agent architecture with any $o_0$-conditioning (a goal encoder, a first-frame feature cache) bypasses the intended difficulty. We report this not as a defect of MIKASA-Robo but as the demonstration the protocol exists for: **the two demand types are distinguishable only by certificate, and benchmarks labeled "memory" mix them.** (The task family is easily repaired — delay the cue onset — and the certificate verifies the repair.)

An exploratory transfer of the §4 comparison to this family (carriers trained on frozen DINO features with a documented transfer recipe, n = 5, direction-only): filter 0.845 (per-seed sd 0.004) vs delta cell 0.600 (sd 0.23) — the direction replicates externally, and so does the delta cell's seed instability.

### 5.2 The instrument audits its authors: encoder deletion is scene-dependent

Our program's prior generation reported "encoder blindness is acquired, not architectural": the task-trained VICReg encoder fails the sighted certificate at low salience ($s^* = $ t1s2; scores 0.297/0.570/0.746 at the rung below) where frozen DINOv2 passes everything. The claim was confounded — architecture, pretraining data, and objective all differed — so we registered a deconfound with the interpretation table frozen in advance: train the *identical* VICReg recipe on a second scene's version of the same salience ladder.

**The correction branch fired.** On the second scene, the task-trained recipe *passes* the rung it fails on the original scene (0.871 / 0.789, floors at chance, health gates passing). The deletion is **scene-dependent** — present where the endogenous scene competes for the objective's variance budget, absent on a sparse scene — and our mechanism attribution is withdrawn as registered. Meanwhile the DINOv2 ladder passes every rung on both scenes at $\geq 0.988$, including a two-pixel, two-frame cue, so $s^*(\mathrm{DINOv2})$ is reported as an *upper bound* ($\leq$ t1s0c), not a threshold: the instrument bottoms out at the scene's render floor before the frozen backbone does.

What survives is the instrument thesis in its strongest form: **whether an encoder deletes your cue is a property of the (encoder, scene) pair that cannot be predicted from the recipe — it must be certified per pair,** and the certificate that told us we were wrong is the same certificate we are proposing.

### 5.3 Delay scaling, and a negative on the obvious repair

The probe-level advantage holds at every tested cue-to-decision delay and shrinks under extrapolation beyond the training length: 0.451 / 0.380 / 0.310 (filter) vs 0.317 / 0.307 / 0.263 (GRU) vs 0.328 / 0.323 / 0.272 (delta cell) at L = 64/96/128 (training length 64, chance 0.25). The natural repair — the filter's spectrum is a design knob, so re-derive its half-lives for the evaluated horizon on frozen weights — was registered with a confirmation bar and **failed it** (+0.002 at L = 128; the L = 64 no-op sanity check returned +0.000 exactly). The decay is not retention-limited (the cue rides an exact eigenvalue-1 hold channel); it is *readout*-limited — the learned weights are fit to length-64 statistics. We report the memory claim scoped to delays near the training regime, with the mechanism for why extrapolation fails attached.

## 6. What the certificates caught: a summary

Across three program generations and one external benchmark, the protocol's catches — each of which would have shipped as a wrong or unscoped claim without it:

| # | catch | certificate | consequence |
|---|---|---|---|
| 1 | tasks that never required memory (prior generation) | integrator floor | benchmark rebuilt |
| 2 | encoders that deleted the cue before memory saw it | sighted | host preflight now mandatory |
| 3 | a −3 to −4 $R^2$ "pathology" that was the readout, not the model | registered probe family | hardest task rejoined the pool, sign unchanged |
| 4 | a world model that cannot roll forward one step | rollout competence (new) | latent-planning claims blocked, substrate renamed |
| 5 | probe rankings that do not predict control rankings (×2) | return-floor | probe-only memory evaluations flagged |
| 6 | an external "memory" task solvable by storing frame 0 | integrator floor | demand-type taxonomy (storage vs filtering) |
| 7 | our own encoder-blindness mechanism claim | second-scene $s^*$ | claim withdrawn, rescoped to per-pair certification |

## 7. Limitations and registered forward work

All confirmatory results live on one scene family plus one external task; the second scene enters only through the $s^*$ grid. The control-level endpoint uses oracle dynamics by measured necessity, and its registered gate is n = 3, direction-only. The envelope's winner fails our representation-health gates (disclosed in §4.2); a delta-cell-appropriate criterion is registered before any architecture headline. $s^*(\mathrm{DINOv2})$ is censored at the render floor. The scene-dependence of encoder deletion is an observation with a registered hypothesis (variance-budget competition), not a mechanism. Forward registrations, frozen in the released record: the variance-relevant uncertainty test on a rollout-competent host; a scene-complexity sweep for the deletion hypothesis; Holm-corrected joint families for any future confirmatory wave.

## 8. Conclusion

Before a memory claim, three questions: does the task demand memory (both sides certified)? did the encoder keep the evidence ($s^*$ for your pair)? does your readout speak for decisions (return-floor, not probes)? Our record shows each question, unasked, has produced a wrong published-grade claim — including two of our own, caught by our own instruments and corrected under frozen registrations in this paper. The certificates are cheap, the instrument is one ladder, and the alternative is a literature of memory results that may be storage results, encoder results, or readout results in disguise.

## Reproducibility statement

Every number in this paper is injected from a hash-manifested artifact by the release renderer; the manuscript manifest binds artifact SHA-256s. The full program record — registrations (timestamped before execution), amendment trails, adversarial review, negative results, and the same-day corrections — ships as the supplement. All experiments ran on three consumer-grade GPUs; the external-benchmark arm requires a documented dependency workaround (a `numpy` pin without Python 3.12 wheels) that blocked three prior attempts and is scripted in the release.

## References

- Becker et al., 2019. Recurrent Kalman Networks. ICML.
- Belinkov, 2022. Probing Classifiers: Promises, Shortcomings, and Advances. CL.
- Cherepanov et al., 2025. MIKASA-Robo: memory-intensive skills assessment for robotics. arXiv:2502.10550.
- Hafner et al., 2019–2023. PlaNet / Dreamer / DreamerV3.
- Hansen et al., 2024. TD-MPC2. ICLR.
- Morad et al., 2023. POPGym. ICLR.
- Pleines et al., 2023. Memory Gym. ICLR.
- Ravichander et al., 2021. Probing the probing paradigm. EACL.
- Samsami et al., 2024. Recall to Imagine (R2I). ICLR.
- Shukor et al. (LeWM lineage), and DINO-WM: cited per the program's V18 reference list.
- Tallec & Ollivier, 2018. Can recurrent neural networks warp time? ICLR.
- von Oswald et al., 2025. MesaNet. arXiv:2506.05233.
- Yang et al., 2024. Gated Delta Networks. arXiv:2412.06464.
- Bardes et al., 2022. VICReg. ICLR. · Oquab et al., 2023. DINOv2. TMLR. · Gu et al., 2020. HiPPO. NeurIPS.
