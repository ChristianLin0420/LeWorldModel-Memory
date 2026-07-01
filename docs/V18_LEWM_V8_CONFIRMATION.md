# V18: frozen stabilized-LeWM + V8 persistent-memory confirmation

## Status and scientific claim

This document freezes the V18 confirmation study before any V18 performance is
inspected. The complete study is:

```text
5 tasks x 8 designs x 5 optimizer seeds x 100 epochs = 200 cells
```

The motivating claim must not be written as “LeWorldModel lacks memory and
causality.” The published LeWorldModel (LeWM) predictor is action-conditioned,
uses temporal causal masking, and attends to a finite observation history. Its
paper uses history length `H=3` for PushT and OGBench-Cube and `H=1` for
TwoRoom. It is therefore temporally causal and has finite-context memory. What
the published architecture does not include is an explicit persistent
recurrent/belief state that survives after an observation leaves that configured
history, and the paper does not test retention or causal use of out-of-window
information under partial observability. See the primary LeWM paper,
Sections 3.1 and D: <https://arxiv.org/html/2603.19312>.

V18 asks the narrower question:

> On previously unopened partially observed dynamics, does compact HACSSM-v8
> add useful persistent state to a healthy, action-conditioned, finite-context
> LeWM-derived pixel JEPA, beyond no memory and matched GRU/SSM recurrent
> carriers; and are V8's action transport and joint access to its two recurrent
> states mechanistically necessary under exact component interventions?

Here “mechanistically necessary” is limited to paired interventions on the
implemented model. It does not mean causal representation learning, causal
identification from observational data, or discovery of environment-level
causal structure.

## Stabilized-host audit and naming boundary

V18 uses LeWM's per-frame pixel encoder, action-conditioned finite-context
predictor, causal action timing, end-to-end next-latent learning, and raw-RGB
deployment interface. It deliberately replaces the original SIGReg
anti-collapse objective with the empirically stabilized active-clean-target
VICReg variance/covariance objective, uses causal encoder normalization, and
trains from paired corrupted-context/clean-target views. Every prediction uses
the true sliding LeWM history of three latent/action tokens; it is not a
one-token decoder disguised by a three-step evaluation burn-in.

That change has a precise consequence:

- It **does not invalidate the within-grid architectural integration test**.
  All eight designs use the same stabilized host, targets, optimizer, data, and
  evaluation. A paired difference therefore isolates the recurrent carrier or
  named V8 intervention within this host.
- It **does invalidate a method-level claim about original LeWorldModel**.
  Original LeWM includes SIGReg as part of the method. V18 cannot be described
  as reproducing its two-term/one-effective-hyperparameter objective, preserving
  its training recipe unchanged, or showing that V8 improves the exact
  SIGReg-based LeWM method.
- The permitted name is **“a stabilized LeWM-derived architecture host”** or
  **“LeWM's encoder/predictor architecture under a VICReg host.”** The shorthand
  “LeWM+V8” is allowed only when this qualification appears at first use. The
  unqualified sentence “V8 improves LeWorldModel” is prohibited.
- V8 still adds no teacher, state label, reward term, hidden-clean update,
  recovery target, horizon loss, or memory-specific loss coefficient. Thus
  “no memory-specific auxiliary objective” remains true; “the LeWM objective is
  unchanged” does not.

A claim about compatibility with or improvement of original SIGReg LeWM needs a
separate exact-SIGReg study whose active targets first pass non-collapse and
convergence gates. V18 cannot supply that external-validity bridge.

## Frozen tasks and trajectory cohort

The five DMC tasks are:

```text
acrobot.swingup
manipulator.bring_ball
quadruped.run
stacker.stack_4
swimmer.swimmer15
```

The architecture, tasks, designs, data seeds, corruption seed, optimizer seeds,
metrics, and decision bars are frozen before opening their V18 results. Each
task receives a newly collected clean cache under independent bounded
tanh-Gaussian actions:

```text
train episodes:       1200
validation episodes:   240
sequence length:         48
RGB resolution:       64 x 64
action process:       IID in native task bounds; smooth_rho=0
train data seed:      270701
validation data seed: 270702
corruption seed:      270711
```

Training corruptions alternate deterministic mean-frame replacement and
spatial cutout over randomized contiguous intervals. Held-out evaluation uses
freeze, Gaussian noise, checkerboard replacement, and longer freeze intervals.
Native DMC task observations and raw physics state are evaluation-only: they
must not enter the encoder, memory, predictor, loss, checkpoint selection, or
early stopping.

## Frozen eight-design grid

All designs use the identical stabilized host. Only the recurrent carrier or
the named V8 intervention changes.

| design | frozen role |
|---|---|
| `vicreg_none` | finite-context host with no persistent recurrent carrier |
| `vicreg_gru` | parameter-matched GRU recurrent baseline |
| `vicreg_ssm` | learned diagonal SSM recurrent baseline |
| `vicreg_hacssmv8` | compact V8 candidate with learned correction shrinkage |
| `vicreg_hacssmv8_noaction` | exact V8 intervention with recurrent action transport zeroed |
| `vicreg_hacssmv8_single` | exact V8 intervention with medium-state-only readout |
| `vicreg_hacssmv8_static` | exact V8 endpoint with correction shrinkage fixed to zero |
| `vicreg_hacssmv8_dynamic` | exact V8 endpoint with correction shrinkage fixed to one |

Compact V8 is the completed SAS-PC architecture: fixed scalar horizons
`tau={2,8}`, one physical shared affine action map, per-level bounded learned
static/dynamic correction shrinkage, a joint RMS-normalized two-state read, and
residual injection into the predictor. `static` and `dynamic` change only the
shrinkage endpoint. `noaction` and `single` are direct mechanism interventions,
not smaller independently redesigned methods. The GRU hidden width is selected
once by the existing parameter-matching rule; it is not tuned per task.

The five optimizer seeds remain frozen as:

```text
{18001, 18002, 18003, 18004, 18005}
```

Every cell trains for exactly 100 epochs with batch size 64 and AdamW
(`lr=3e-4`, `weight_decay=1e-5`), embedding width 128, and predictor history 3.
There is no result-dependent early stopping, checkpoint cherry-picking,
architecture revision, task exclusion, seed exclusion, or rescue sweep. All
200 cells must finish irrespective of intermediate outcomes.

## Four-GPU execution schedule

Cells are serial within each task queue. The fixed schedule is:

| GPU | sequential task queue | cells |
|---|---|---:|
| `0` | `acrobot.swingup`, then `stacker.stack_4` | 80 |
| `1` | `manipulator.bring_ball` | 40 |
| `2` | `quadruped.run` | 40 |
| `3` | `swimmer.swimmer15` | 40 |

GPU 0 must not run Acrobot and Stacker concurrently. Its second task begins only
after all 40 Acrobot cells have valid terminal receipts. The other three queues
may run concurrently with GPU 0. A task's design/seed cells run one at a time to
avoid within-GPU resource and timing confounds.

Each cell must produce its checkpoint, 100-row metrics history, metrics JSON,
paired held-out rollout NPZ/video/table, command and source hashes, and finished
online W&B receipt. The runner must be cell-resumable but may never overwrite a
valid completed cell. Launch also requires the complete write-once ten-cache
cohort manifest and its SHA-256 sidecar. Source hashes are rechecked before
every cell; cache identity is checked before each task and again before final
analysis. Disabled W&B or skipped final analysis is not a valid confirmation
mode.

### Implementation audit before launch

The V18 data adapter, trainer, runner, and analyzer must agree exactly with this
document before collection. The audited implementation uses a scoped V11 cache
schema adapter, true sliding `H=3` latent/action windows, active clean targets,
the eight frozen designs, the four fixed GPU queues, write-once source/cache/
command hashes, and a 100,000-draw crossed analyzer. Focused tests cover every
registry, window/target alignment, GRU pre-observation causality, the 200-cell
command expansion, per-cell comparator envelopes, all success gates, and
write-once outputs. Launch is permitted only after those tests pass and the
exact source is committed; any subsequent source edit invalidates resume under
the stored source hashes.

## Metrics and fixed contrasts

The primary metric is the equal-task, equal-seed paired reduction in held-out
**prior task-state NMSE**, averaged over the four held-out corruptions. The prior
is measured before current-observation correction, so it targets information
transported through missing observations. Raw state dimensions and unnormalized
MSE are never pooled across tasks.

Secondary reports include visible next-clean-latent prediction, clean prior
NMSE, per-corruption gap/deep/first-post prior NMSE, active-target and encoder
variance/rank, late-window convergence, V8 gate/route telemetry, and the
checkpoint-matched legal initial-frame-plus-action integrator.

For task `t` and optimizer seed `s`, let lower error be better and define:

- **better recurrent reference:** the per-cell lower held-out prior NMSE of the
  separately trained `vicreg_gru` and `vicreg_ssm` cells;
- **static/dynamic envelope:** the per-cell lower held-out prior NMSE of
  `vicreg_hacssmv8_static` and `vicreg_hacssmv8_dynamic`;
- **paired relative reduction:** `(reference - V8) / max(abs(reference), eps)`;
- **task win:** positive reduction after averaging candidate and reference
  errors over the five optimizer seeds within that task.

The better-recurrent and endpoint-envelope definitions are fixed and may not be
replaced after seeing whether GRU, SSM, static, or dynamic wins a particular
cell. They deliberately give V8 the stricter per-cell comparator.
The recurrent identity is selected once from the primary held-out prior NMSE
and that same model supplies the deep-gap and clean-prior reference values for
clauses 5 and 9; those clauses do not select a second metric-specific oracle.
An exact primary tie selects `vicreg_gru` (lexicographic design-ID tie break).

Crossed uncertainty uses 100,000 percentile-bootstrap draws. Task indices and
seed indices are independently resampled with replacement and combined by their
Cartesian product; the statistic is the equal-task/equal-seed mean paired
relative reduction. Use `numpy.random.Generator(PCG64(18018))` and linear
2.5/97.5 percentiles for the 95% interval. All intervals and all 25 cell effects
must be reported even where a gate does not require them.

## Exact ICLR success bars

The positive label `STABILIZED_LEWM_V8_CONFIRMATION_PASS` is conjunctive. It
requires every clause below; rounded display values cannot pass a threshold
that the full-precision value misses.

1. **Integrity:** exactly `200/200` cells are finite and artifact-valid, with
   complete histories and finished W&B/source/cache/command receipts.
2. **V8 versus the better recurrent reference:** mean paired reduction is at
   least `3%`; the crossed task-by-seed 95% CI lower bound is strictly above
   zero; at least `18/25` cell effects are positive; and at least `4/5` task
   effects are positive.
3. **V8 versus no persistent carrier:** mean paired reduction versus
   `vicreg_none` is at least `5%`, with at least `20/25` positive cells and at
   least `4/5` positive tasks.
4. **Legal-integrator guard:** V8 improves on its own checkpoint-matched legal
   initial-frame-plus-action integrator by at least `3%`, with at least `18/25`
   positive cells and at least `4/5` positive tasks.
5. **Deep-gap persistence:** on the equal-cell deep-gap prior metric, V8's
   crossed 95% CI lower bound versus the same per-cell better GRU/SSM reference
   is strictly above zero, with at least `3/5` positive task effects.
6. **Action transport:** V8 improves on `vicreg_hacssmv8_noaction` by at least
   `5%`, the crossed 95% CI lower bound is strictly above zero, at least `18/25`
   cell effects are positive, and at least `4/5` task effects are positive.
7. **Joint-state access:** V8 improves on `vicreg_hacssmv8_single` by at least
   `3%`, the crossed 95% CI lower bound is strictly above zero, at least `18/25`
   cell effects are positive, and at least `4/5` task effects are positive.
8. **Learned-shrinkage noninferiority:** V8's mean degradation relative to the
   per-cell static/dynamic endpoint envelope is at most `1%` (equivalently,
   mean paired relative reduction is at least `-1%`), and the crossed 95% CI
   lower bound is at least `-1%`. This licenses only noninferiority, not
   superiority or learned adaptation.
9. **Clean-state guard:** mean clean-prior NMSE degradation versus the same
   per-cell better GRU/SSM reference is at most `3%`.
10. **Representation health:** all `200/200` cells have finite representation
    metrics, encoder mean-channel variance at least `1e-4`, and encoder
    covariance effective rank at least `16`. A collapsed comparator cannot
    support a V8 superiority claim.
11. **Convergence:** all `200/200` cells have finite absolute relative
    late-window predictive-loss change at most `5%`, comparing epochs 81--90
    with 91--100. A nonconverged comparator likewise invalidates superiority.

Failure of any clause gives `CONFIRMATION_FAILED` while preserving and reporting
the complete 200-cell result. Missing, duplicate, nonfinite, overwritten, or
unverifiable cells give `INCOMPLETE_OR_INVALID`. Neither label authorizes a
threshold change, task deletion, new seed, architecture revision, or rerun on
this cohort. Static, dynamic, GRU, and SSM results remain direct results, not a
post-hoc explanation omitted from the paper.

## Paper claim boundary

If every bar passes, the strongest permitted conclusion is:

> Across five previously unopened DMC corruption cohorts, compact V8 adds useful
> persistent state to a stabilized VICReg-trained LeWM-derived finite-context
> pixel JEPA, outperforming no recurrent carrier and a conservative per-cell
> GRU/SSM reference on evaluation-only prior-state prediction. Exact model
> interventions show that recurrent action transport and joint two-state access
> are necessary for the measured V8 effect in this implementation.

Even after a pass, V18 does not license claims of:

- improvement to the exact SIGReg-based LeWorldModel method or preservation of
  LeWM's original objective;
- absence of temporal causality or all memory in published LeWM;
- causal discovery, causal representations, or identified environment action
  effects;
- first persistent, recurrent, action-conditioned, multi-timescale, or
  predict--correct world model;
- improved planning, executed return, policy quality, calibrated uncertainty,
  semantic hierarchy, learned timescale discovery, or robustness outside these
  corruptions and tasks;
- superiority of learned shrinkage: clause 8 establishes noninferiority to an
  oracle-like per-cell endpoint envelope, not that learning selects the correct
  endpoint or beats it.

If V18 fails, the finite-context observation about published LeWM remains an
architectural fact, but V8 is not a confirmed repair. The result must be framed
as a complete falsification/mechanism study with task heterogeneity and direct
controls, not rescued by adaptive revisions on this cohort.
