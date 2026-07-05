# V21 submission review — is this an ICLR paper? (2026-07-05)

**Method.** Three independent adversarial reviewer lenses (rigor/claims-evidence, significance/positioning, external-validity/baselines) run over the completed V21 record (`docs/V21_PROPOSAL.md` §1–§11 + artifacts), synthesized here with the execution-level provenance only the program log knows (registered vs post-registration status of each number). Every artifact-level allegation raised by the panel was independently re-verified before inclusion; two led to same-day repairs (below).

**Scores.** Rigor 5/10 · Significance 5/10 · External validity 4/10 (as architecture claim) / 6 (as methodology). Synthesis: **not submittable as the architecture paper (Paper B ≈ 3–4, do not submit); submittable as Paper A — the methodology/certification paper — at a current 6, with a credible 7–8 after the ranked fixes below,** of which the first is priced at under a week of GPU time.

---

## 1. Where the three lenses converged (treat as ground truth)

**C1 — The consumer never consumed the world model.** All three lenses, independently: X2's executed "return-level" result is a frozen linear selector probe at t=24 feeding an oracle-physics planner. Arithmetically, rfix success 0.769 ≈ selector accuracy 0.825 × oracle ceiling 0.917 — decision-thresholding of probe accuracy, not planning with a belief. X2-Finding-1 (the host cannot roll forward one useful step) is honest and valuable, but it means the paper cannot say "world-model planning" anywhere. Required rename: **belief-conditioned goal selection under oracle execution.** Claim 4 stays real under that name; the "×3 amplification" framing goes.

**C2 — Nothing ever leaves the one scene.** Three consecutive program generations registered an external arm (V19 §4.4, V20 I5, V21 §4/X3) and none ran it; the record's own feasibility note prices the fix at 0.5–1 day. t1/t3/t4 are sibling overlays on one DMC scene. The panel is unanimous that a third registration-without-execution is no longer presentable: **one external task through the pipeline is close to a hard precondition for either paper.** Even an *informative certification failure* on MIKASA-Robo counts — the certificates are the product.

**C3 — Claim 6 as registered is predominantly unmet.** Leg (i) external family: never tested. Leg (ii) second-host s\*: passed but left-censored (DINOv2 saturates every rung — no threshold localized) and three-way confounded (architecture, data, objective all differ), so "acquired, not architectural" is one of ≥3 live explanations. Leg (iii): the registered confirmed-if was "advantage **grows** with delay"; measured, it shrinks monotonically (+0.134 → +0.073 → +0.047). §11's prose is honest about each piece; the §5 ladder row must nevertheless be reported **unmet**, and the words "portable"/"general" cannot appear attached to the result — only to the instruments, and even that weakly.

**C4 — The fair envelope has two cracks.** (a) Verified in this audit: all 30 `gdelta_l10` X1 cells fail the program's registered health gates (effective rank 3–11 vs 16 minimum) while lkc_rfix passes 27/30 — either the rank gate doesn't transfer to matrix-state cells (then X0a's healthy-only sensitivity is arm-relative) or the confirmed rival is a degenerate trainee. Disclosed in §9 as of today; unadjudicated. (b) The envelope-arm control result (gdelta 0.339 < acgru 0.508, exploratory) shows probe-dev selection does not predict control ranking — the envelope was selected on the wrong coordinate for the control claim. (c) No transformer/long-context baseline and no RKN lineage despite §7 naming it — table stakes for a 2026 memory-architecture paper. All three cracks bear on Paper B far more than Paper A.

**C5 — The registered multiplicity structure was abandoned.** The X1–X2 Holm family became per-family gates ("superseded by the amendment trail"); X2 has no registered α and n=3 with no CI (sign test floors at p=0.125); claim 5's registered confirmed-if (σ-aware value on variance-relevant segments, probes-can't-tell) was neither met nor testable after T4-act's de-scope — its "mechanism split" is a legitimate finding but **exploratory**, not confirmatory. The confirmatory layer of any submission is exactly: claim 1, claim 2 (with C4a disclosed), claim 3, and claim 4's registered acgru gate. Nothing else.

## 2. Same-day repairs made during this review

1. **Claim-5 mediation statistic had no artifact** (0.825 → 0.536 lived only in prose). Recomputed deterministically from frozen checkpoints; exact reproduction; persisted to `outputs/v21_x2/x2_selector_stats.json` (`scripts/x2_selector_stats_v21.py`).
2. **Health-gate asymmetry disclosed** in §9 with both readings on the record.
3. Claim-4 envelope addendum, T4-act de-scope rationale, and the X0b/X1 execution record were written into the proposal earlier today (they existed only as artifacts).

## 3. Verdict

**V21 is one experiment short of a submittable Paper A and one full wave short of a defensible Paper B.**

- **Paper A** (certification methodology: demand certificates + s\* + probe/control dissociation, with the preregistered inversion as the worked case study, claims 5/6 reported as exploratory/unmet): currently a 6. The panel's unanimous flip condition is **contact with any world the program did not build** — one MIKASA-Robo task (RememberColor) through the certification pipeline. With that plus the honest rescoping already in the doc, consensus estimate **7, borderline 8**. Do not re-merge: C1–C4 all price Paper B's headline, and the re-merge would import them.
- **Paper B** (architecture claim): blocked by C4 (sick rival + wrong selection coordinate + missing transformer baseline) and C3 (delay direction). Needs a new confirmation wave — health-symmetric envelope at n=10 including an attention control, a single pre-registered Holm family spanning probe and control endpoints, X2 at ≥10 seeds with a registered test — and the external arm. ≈ 2–3 weeks of GPU time. Park it.

## 4. Ranked fix list (cost → score movement)

| # | Fix | Cost | Moves |
|---|---|---|---|
| 1 | MIKASA-Robo RememberColor through certification (dedicated venv; even a failed certificate is reportable) | 0.5–1 day eng + ~1 GPU-day | A: 6→7 (all three lenses name it) |
| 2 | Extend the DINOv2 s\* ladder downward until it fails (threshold, not censoring); one deconfounding host (VICReg-recipe off-stream, or on-stream-adapted DINOv2) | ~0.5–1 GPU-day | A: firms the s\*-instrument leg; removes C3(ii) |
| 3 | Delay point with the filter's spectrum set for the tested delay (the horizon knob) — shows the delay law is design-controllable | ~0.5 GPU-day, frozen checkpoints | A appendix; partially rehabilitates C3(iii) |
| 4 | Delta-cell-appropriate health criterion, registered; or rank-passing delta config at n=10 | 0–2 GPU-days | B precondition; A can ship with the disclosure |
| 5 | Transformer/long-context + ac-RKN envelope members through X1's frozen gate | ~2–3 GPU-days | B only |
| 6 | X2 at ≥10 checkpoint seeds with registered test + α; selector stats persisted (done for mediation) | ~1 GPU-day | B; A cites current n=3 as case study honestly |

**Recommended sequence for an ICLR deadline: fixes 1–3 (≈ 2–3 GPU-days total), then write Paper A.** Fixes 4–6 are the V22 program if Paper B is wanted later.
