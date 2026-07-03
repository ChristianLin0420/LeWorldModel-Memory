# Proposed post-V18 edits for `docs/LEARNABLE_MEMORY.md`

This is a staging document only. It does not modify the tracked evidence
record. The release renderer fills every token from the validated, write-once
V18 analysis, PDF check, and provenance-bound review bundle and refuses any
leftover token before this document may be applied.

Placeholder conventions used below:

- `COMPLETE`, `CONFIRMATION_FAILED`,
  `false`,
  `200`, and
  `PASS` come directly from
  `confirmation_analysis.json`.
- `{{R_*}}`, `{{N_*}}`, `{{I_*}}`, `{{D_*}}`, `{{A_*}}`, `{{J_*}}`,
  `{{E_*}}`, and `{{C_*}}` denote the registered recurrent-envelope,
  no-carrier, legal-integrator, deep-gap, no-action, single-read,
  endpoint-envelope, and clean-prior contrasts, respectively. `MEAN` is the
  mean paired relative reduction; `CI` is the crossed-bootstrap 95% interval;
  `WINS` and `TASKS` are positive cell and task effects; and `GATE` is the
  analyzer's exact gate result.
- Representation, convergence, protocol, artifact, CSV, and manuscript hashes
  must be copied from the analyzer/manifests, never recomputed from a different
  bundle or inferred from displayed values.
- `The registered conjunction fails, so SAS-PC/V8 is not confirmed as a generally superior persistent carrier; favorable individual contrasts cannot rescue the frozen decision.` and `The submission is a complete frozen falsification: SAS-PC/V8 is not confirmed as a generally superior persistent carrier, and favorable subsets do not override the failed conjunction.` must be rendered
  mechanically from the scientific label. A failed conjunction must say that
  SAS-PC/V8 is not confirmed as a generally superior persistent carrier;
  favorable individual contrasts cannot rescue it.

## Block 1 — replace the opening italicized status paragraph

Replace the complete italicized paragraph immediately below the document title
with this block:

```markdown
*Design and experiment record. V1–V11 live in
`lewm/models/memory.py`; SIRO-v12, CF-HIRO-v13, CF-EBO-v14, and CVPF-v15 use
their isolated model modules and the common LeWM host. Status as of
2026-07-02: V1–V9 are complete with their recorded negative
pilot/final labels; V10-J has no official launch; V11's excluded screen is
negative; V12–V14 are complete `SCREEN_NO_GO` studies; and V15 remains
`INCOMPLETE_OR_INVALID / FAIL_CLOSED`. V16 completed 144/144 opened-cache
host-objective cells and found Sub-JEPA collapse; V17 completed 72/72 cells
with label `ADAPTIVE_COLLAPSE_REPAIR_FAILED`. V18 completed
200/200 artifact-valid cells on five previously
unopened raw-pixel tasks and the write-once analyzer returned
`CONFIRMATION_FAILED`; `official_confirmation_result` is
`false` (§7.18–§7.20). V18 tests unchanged
compact V8 inside a causally normalized, active-clean-target, VICReg-trained
LeWM-derived host with a true sliding three-token predictor. It is not a test
of the exact original SIGReg LeWorldModel objective and contains no executed
return, success, control, or planning evaluation. Review-safe result receipts
and the anonymous paper are linked in §7.20 and §10.*
```

## Block 2 — append to the empirical verdict

Append this paragraph after the existing empirical-verdict paragraph ending
with V15's `INCOMPLETE_OR_INVALID / FAIL_CLOSED / NO_100E_LAUNCH` sentence:

```markdown
V16 and V17 then isolate the host objective rather than rescuing a memory
method: V16 collapses, while V17 repairs rank and variance but fails its frozen
convergence/effect decision. V18 is the first new-task, true-`H=3`,
artifact-complete confirmation of unchanged compact V8 in this sequence. Its
write-once result is `CONFIRMATION_FAILED` over
200/200 valid cells: SAS-PC changes held-out
prior-state NMSE by -2.10% versus the per-cell better GRU/SSM reference
(95% CI [-13.35%, +9.11%]; 11/25 cell and 2/5 task wins) and by
+12.44% versus no persistent carrier. The registered conjunction fails, so SAS-PC/V8 is not confirmed as a generally superior persistent carrier; favorable individual contrasts cannot rescue the frozen decision. This result
concerns persistent information transport and exact action-transport/joint-read
interventions inside a stabilized VICReg LeWM-derived host. It neither shows
that published LeWM lacks temporal causality or all memory nor establishes
improvement to exact SIGReg LeWM, causal discovery, executed return, or
planning (§7.20).
```

## Block 3 — replace §7.20 in full

Replace everything from the current `### 7.20` heading through the paragraph
ending “without changing this prospective contract” with this block:

````markdown
### 7.20 LeWM+V8-v18: complete unopened-task confirmation (`CONFIRMATION_FAILED`)

V18 is the first prospective test in this sequence that combines unchanged
compact SAS-PC/V8 with a true sliding three-token LeWM predictor on new
raw-pixel DMC tasks. The question is deliberately narrower than “LeWM lacks
memory or causality.” Published LeWorldModel is action-conditioned and
temporally causal over a configured finite observation history (`H=3` for
PushT and OGBench-Cube and `H=1` for TwoRoom). It does not include an explicit
persistent recurrent/belief state that survives after evidence leaves that
window. V18 asks whether the added carrier transports useful out-of-window
information under partial observability.

The frozen grid remained:

```text
tasks:    acrobot.swingup, manipulator.bring_ball, quadruped.run,
          stacker.stack_4, swimmer.swimmer15
designs:  vicreg_{none,gru,ssm,hacssmv8,hacssmv8_static,
                  hacssmv8_dynamic,hacssmv8_noaction,hacssmv8_single}
seeds:    {18001,18002,18003,18004,18005}
budget:   100 epochs
total:    5 tasks x 8 designs x 5 seeds = 200 cells
```

Every arm uses the same end-to-end RGB encoder, aligned `H=3` latent/action
windows, active synchronized clean target, and VICReg variance/covariance
stabilization. V8 receives no teacher, reward/state target, hidden-clean
update, memory-specific loss, selectable coefficient, or task-specific
setting. `vicreg_none` retains the finite LeWM context but has no persistent
carrier; GRU and diagonal SSM are separately trained recurrent references;
static/dynamic form a conservative endpoint envelope; and no-action and
single-read are exact, separately trained mechanism interventions. Native task
observations and physics state are evaluation-only.

The write-once analyzer validated **200/200** cells
and returned:

| receipt | analyzer value |
|---|---|
| status | `COMPLETE` |
| scientific label | **`CONFIRMATION_FAILED`** |
| official confirmation result | `false` |
| artifact integrity | `PASS` |
| primary endpoint | held-out pre-observation task-state NMSE |

| registered comparison or validity guard | observed effect / receipt | 95% CI | cell wins | task wins | gate |
|---|---:|---:|---:|---:|---:|
| SAS-PC vs per-cell better GRU/SSM | -2.10% | [-13.35%, +9.11%] | 11/25 | 2/5 | **FAIL** |
| SAS-PC vs no persistent carrier | +12.44% | [+3.65%, +22.43%] | 24/25 | 5/5 | **PASS** |
| SAS-PC vs legal initial-frame/action integrator | -38.12% | [-61.38%, -20.58%] | 0/25 | 0/5 | **FAIL** |
| deep-gap persistence vs primary-selected GRU/SSM | -1.74% | [-12.95%, +9.46%] | 11/25 | 2/5 | **FAIL** |
| recurrent action-transport intervention | +9.48% | [+2.88%, +16.09%] | 23/25 | 5/5 | **PASS** |
| joint two-state-read intervention | -5.39% | [-13.12%, +0.67%] | 6/25 | 1/5 | **FAIL** |
| learned shrinkage vs per-cell endpoint envelope | -1.89% | [-4.88%, +0.37%] | 6/25 | 0/5 | **FAIL** |
| clean-prior guard vs primary-selected GRU/SSM | +11.73% | [+5.82%, +17.70%] | 25/25 | 5/5 | **PASS** |
| representation health | min variance 0.0228; min rank 2.02 | — | variance 200/200; rank 144/200 | — | **FAIL** |
| convergence | max absolute late change +132.09% | — | 126/200 converged | — | **FAIL** |

**Registered decision.** The registered conjunction fails, so SAS-PC/V8 is not confirmed as a generally superior persistent carrier; favorable individual contrasts cannot rescue the frozen decision. The decision is the frozen
conjunction of integrity, recurrent/no-carrier/integrator comparisons,
deep-gap persistence, action-transport and joint-read interventions,
endpoint-envelope noninferiority, clean-state quality, representation health,
and convergence. A favorable subset does not override a failed clause, and no
threshold, task, seed, or architecture is changed after opening the cohort.

The intervention language is deliberately local. The no-action comparison
tests recurrent action transport in this implementation; the single-read
comparison tests access to both recurrent states. Neither identifies
environment-level causal structure, discovers causal variables, or establishes
causal representation learning.

V18 uses LeWM's encoder/predictor architecture under a causally normalized,
active-clean-target **VICReg host**. It can therefore support or falsify a
persistent-state integration claim within that shared host, but it cannot be
described as improving the exact original SIGReg-based LeWorldModel method or
preserving LeWM's original two-term objective. No policy is executed, so V18
also supplies no return, success, control, or planning evidence.

The final public record is provenance-bound across the
[frozen protocol](V18_LEWM_V8_CONFIRMATION.md),
[write-once analysis](../paper/review_artifact/confirmation_analysis.json),
[200-cell table](../paper/review_artifact/confirmation_cells.csv),
[registered contrasts](../paper/review_artifact/confirmation_contrasts.csv),
[redacted execution receipts](../paper/review_artifact/confirmation_runs.redacted.json),
and [review manifest](../paper/review_artifact/review_manifest.json). The
[analysis-rendered manuscript](ICLR.md),
[manuscript manifest](ICLR.manifest.json), and
[anonymous PDF](../paper/main.pdf) report the same scientific label and hashes.
The review bundle excludes private checkpoints, rollout arrays, raw histories,
and identity-bearing remote metadata; their aggregate identity remains bound
by the public receipts.
````

## Block 4 — replace the §10 opening and scope the old audit historically

Replace the existing §10 heading and its immediate `### Decision` heading with
the following block. Leave the old prose beginning “There is a worthwhile
controlled finding here” immediately after this block.

```markdown
## 10. ICLR submission record

### 10.1 Current V18 submission status (`CONFIRMATION_FAILED`)

The current anonymous manuscript,
[*Finite Context Is Not Persistent State: A Frozen Falsification Study in a
LeWorldModel-Derived JEPA*](ICLR.md), is a new V18 paper rather than a revision
of the pre-V18 two-timescale manuscript audited below. It reports the complete
frozen five-task, eight-design, five-seed study and is bound to
`CONFIRMATION_FAILED` by the
[manuscript manifest](ICLR.manifest.json),
[review-artifact manifest](../paper/review_artifact/review_manifest.json), and
[write-once decision](../paper/review_artifact/confirmation_analysis.json).
The [anonymous PDF](../paper/main.pdf) contains 15 pages,
with 9 pages of main text under the public ICLR 2026 style
used for the current format check; the ICLR 2027 package and final author guide
must still be substituted and rechecked when officially available. Rebuild and
template-version details are in the [paper README](../paper/README.md).

**Current framing.** The submission is a complete frozen falsification: SAS-PC/V8 is not confirmed as a generally superior persistent carrier, and favorable subsets do not override the failed conjunction. The strongest permissible
claim is confined to persistent prior-state information transport and named
component interventions in a stabilized VICReg-trained LeWM-derived host on
the frozen corruption cohort. Published LeWM remains correctly described as
action-conditioned, temporally causal, and finite-context; the study does not
test exact SIGReg LeWM. It contains no executed policy, return, success, or
planning evaluation and makes no causal-discovery, causal-representation,
learned-timescale, semantic-hierarchy, or calibrated-uncertainty claim.

Release status: **FORMAT_CHECK_COMPLETE_UNDER_OFFICIAL_ICLR_2026_STYLE; SUBMISSION_BLOCKED_PENDING_OFFICIAL_ICLR_2027_TEMPLATE_AND_FINAL_AUTHOR_GUIDE**. This status requires an internally
consistent 200-cell review bundle, no unresolved result placeholders, matching
analysis/manuscript/figure hashes, no identity leak, a clean PDF build, no
undefined citations or serious overfull boxes, and compliance with the final
ICLR 2027 format and submission rules. Scientific completion alone does not
waive those release checks.

### 10.2 Historical pre-V18 audit and recommendation (2026-06-30)

Everything from the historical decision below through the end of §10 audits
the former 17-page two-timescale-memory manuscript and evidence only through
V15. It predates the V16/V17 host audits, the frozen V18 cohort, the current
analysis-rendered manuscript, and the review-safe result bundle. Its criticisms
remain part of the research record, but phrases such as “the current PDF,” “do
not submit the current manuscript,” and “a confirmation successor is still
required” refer to that superseded pre-V18 manuscript. They must not be read as
descriptions of the V18 paper above.

### 10.3 Historical decision: do not submit the pre-V18 manuscript to the ICLR main track as written
```

For consistent historical scope, replace the four later §10 headings exactly
as follows; their body text stays unchanged:

```markdown
### 10.4 Historical blocking scientific issues

### 10.5 Historical novelty and positioning risk

### 10.6 Historical manuscript readiness

### 10.7 Historical defensible submission path
```

The old final recommendation should remain visibly historical. Prefix its
existing paragraph with this sentence:

```markdown
**Historical recommendation for the superseded pre-V18 manuscript.** The
following recommendation was recorded before V18 and is retained without
retroactively treating V18 as executed-return evidence:
```
