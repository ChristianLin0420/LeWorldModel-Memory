# Proposed post-V18 edits for `docs/LEARNABLE_MEMORY.md`

This is a staging document only. It does not modify the tracked evidence
record. The release renderer fills every token from the validated, write-once
V18 analysis, PDF check, and provenance-bound review bundle and refuses any
leftover token before this document may be applied.

Placeholder conventions used below:

- `{{V18_STATUS}}`, `{{V18_SCIENTIFIC_LABEL}}`,
  `{{V18_OFFICIAL_CONFIRMATION_RESULT}}`,
  `{{V18_COMPLETED_VALID_CELLS}}`, and
  `{{V18_ARTIFACT_INTEGRITY}}` come directly from
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
- `{{V18_DECISION_SENTENCE}}` and `{{V18_SUBMISSION_FRAMING}}` must be rendered
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
{{V18_FINAL_DATE}}: V1–V9 are complete with their recorded negative
pilot/final labels; V10-J has no official launch; V11's excluded screen is
negative; V12–V14 are complete `SCREEN_NO_GO` studies; and V15 remains
`INCOMPLETE_OR_INVALID / FAIL_CLOSED`. V16 completed 144/144 opened-cache
host-objective cells and found Sub-JEPA collapse; V17 completed 72/72 cells
with label `ADAPTIVE_COLLAPSE_REPAIR_FAILED`. V18 completed
{{V18_COMPLETED_VALID_CELLS}}/200 artifact-valid cells on five previously
unopened raw-pixel tasks and the write-once analyzer returned
`{{V18_SCIENTIFIC_LABEL}}`; `official_confirmation_result` is
`{{V18_OFFICIAL_CONFIRMATION_RESULT}}` (§7.18–§7.20). V18 tests unchanged
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
write-once result is `{{V18_SCIENTIFIC_LABEL}}` over
{{V18_COMPLETED_VALID_CELLS}}/200 valid cells: SAS-PC changes held-out
prior-state NMSE by {{R_MEAN}} versus the per-cell better GRU/SSM reference
(95% CI {{R_CI}}; {{R_WINS}}/25 cell and {{R_TASKS}}/5 task wins) and by
{{N_MEAN}} versus no persistent carrier. {{V18_DECISION_SENTENCE}} This result
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
### 7.20 LeWM+V8-v18: complete unopened-task confirmation (`{{V18_SCIENTIFIC_LABEL}}`)

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

The write-once analyzer validated **{{V18_COMPLETED_VALID_CELLS}}/200** cells
and returned:

| receipt | analyzer value |
|---|---|
| status | `{{V18_STATUS}}` |
| scientific label | **`{{V18_SCIENTIFIC_LABEL}}`** |
| official confirmation result | `{{V18_OFFICIAL_CONFIRMATION_RESULT}}` |
| artifact integrity | `{{V18_ARTIFACT_INTEGRITY}}` |
| primary endpoint | held-out pre-observation task-state NMSE |

| registered comparison or validity guard | observed effect / receipt | 95% CI | cell wins | task wins | gate |
|---|---:|---:|---:|---:|---:|
| SAS-PC vs per-cell better GRU/SSM | {{R_MEAN}} | {{R_CI}} | {{R_WINS}}/25 | {{R_TASKS}}/5 | **{{R_GATE}}** |
| SAS-PC vs no persistent carrier | {{N_MEAN}} | {{N_CI}} | {{N_WINS}}/25 | {{N_TASKS}}/5 | **{{N_GATE}}** |
| SAS-PC vs legal initial-frame/action integrator | {{I_MEAN}} | {{I_CI}} | {{I_WINS}}/25 | {{I_TASKS}}/5 | **{{I_GATE}}** |
| deep-gap persistence vs primary-selected GRU/SSM | {{D_MEAN}} | {{D_CI}} | {{D_WINS}}/25 | {{D_TASKS}}/5 | **{{D_GATE}}** |
| recurrent action-transport intervention | {{A_MEAN}} | {{A_CI}} | {{A_WINS}}/25 | {{A_TASKS}}/5 | **{{A_GATE}}** |
| joint two-state-read intervention | {{J_MEAN}} | {{J_CI}} | {{J_WINS}}/25 | {{J_TASKS}}/5 | **{{J_GATE}}** |
| learned shrinkage vs per-cell endpoint envelope | {{E_MEAN}} | {{E_CI}} | {{E_WINS}}/25 | {{E_TASKS}}/5 | **{{E_GATE}}** |
| clean-prior guard vs primary-selected GRU/SSM | {{C_MEAN}} | {{C_CI}} | {{C_WINS}}/25 | {{C_TASKS}}/5 | **{{C_GATE}}** |
| representation health | min variance {{V18_MIN_VARIANCE}}; min rank {{V18_MIN_RANK}} | — | variance {{V18_VARIANCE_PASSING_CELLS}}/200; rank {{V18_RANK_PASSING_CELLS}}/200 | — | **{{V18_REPRESENTATION_GATE}}** |
| convergence | max absolute late change {{V18_MAX_LATE_CHANGE}} | — | {{V18_CONVERGED_CELLS}}/200 converged | — | **{{V18_CONVERGENCE_GATE}}** |

**Registered decision.** {{V18_DECISION_SENTENCE}} The decision is the frozen
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

### 10.1 Current V18 submission status (`{{V18_SCIENTIFIC_LABEL}}`)

The current anonymous manuscript,
[*Finite Context Is Not Persistent State: A Frozen Falsification Study in a
LeWorldModel-Derived JEPA*](ICLR.md), is a new V18 paper rather than a revision
of the pre-V18 two-timescale manuscript audited below. It reports the complete
frozen five-task, eight-design, five-seed study and is bound to
`{{V18_SCIENTIFIC_LABEL}}` by the
[manuscript manifest](ICLR.manifest.json),
[review-artifact manifest](../paper/review_artifact/review_manifest.json), and
[write-once decision](../paper/review_artifact/confirmation_analysis.json).
The [anonymous PDF](../paper/main.pdf) contains {{V18_PDF_TOTAL_PAGES}} pages,
with {{V18_PDF_MAIN_PAGES}} pages of main text under the public ICLR 2026 style
used for the current format check; the ICLR 2027 package and final author guide
must still be substituted and rechecked when officially available. Rebuild and
template-version details are in the [paper README](../paper/README.md).

**Current framing.** {{V18_SUBMISSION_FRAMING}} The strongest permissible
claim is confined to persistent prior-state information transport and named
component interventions in a stabilized VICReg-trained LeWM-derived host on
the frozen corruption cohort. Published LeWM remains correctly described as
action-conditioned, temporally causal, and finite-context; the study does not
test exact SIGReg LeWM. It contains no executed policy, return, success, or
planning evaluation and makes no causal-discovery, causal-representation,
learned-timescale, semantic-hierarchy, or calibrated-uncertainty claim.

Release status: **{{V18_RELEASE_STATUS}}**. This status requires an internally
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
