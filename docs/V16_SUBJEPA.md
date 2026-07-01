# V16 paired-view Sub-JEPA: adaptive development protocol

## Status and scope

V16 is a deliberately small host-objective study. It combines the existing causal LeWorldModel training path with Multi-Subspace SIGReg from [Sub-JEPA](https://arxiv.org/abs/2605.09241), then crosses that host with only three already implemented memory choices: no recurrent memory, the learned diagonal SSM, and compact HACSSM-v8. It does not introduce another observer, filter, hierarchy, correction rule, auxiliary dynamics model, teacher, or planning objective.

The complete experiment is fixed at

```text
4 host regularizers x 3 memory backbones x 4 tasks x 3 optimizer seeds
= 12 designs x 4 tasks x 3 seeds
= 144 independently trained 30-epoch cells
```

This is an **adaptive development grid on already opened caches**. It is not a confirmation experiment, an official continuation of V15, evidence of out-of-sample architectural selection, or executed-return/control evidence. No result from this grid may be promoted by renaming it a pilot or confirmation run.

## Why this method

[LeWorldModel](https://arxiv.org/abs/2603.19312) trains an end-to-end JEPA with next-embedding prediction and full-space SIGReg. [Sub-JEPA](https://arxiv.org/abs/2605.09241) changes only the anti-collapse term: it projects the latent into multiple fixed, row-orthonormal random subspaces and applies SIGReg in each projected view. Its [released reference implementation](https://github.com/intcomp/Sub-JEPA) adds no deployed state and leaves the encoder, predictor, prediction target, and planner unchanged.

The source paper reports six-seed planning success of `95.00/84.00/89.00/76.33` on Two-Room/Reacher/PushT/OGB-Cube, versus its LeWM rerun at `84.33/82.67/84.67/67.33`. Those numbers motivate a test; they do not establish that the method transfers to this repository's causal normalization, paired corruption views, `D=128` representation, or partially observed memory protocol. The source also selected `K=32` for three tasks and `K=16` for PushT after validation ablations. Its own `K=32` PushT result falls to `28%`, so V16 freezes `K=16` as the primary configuration and treats `K=32` only as an aggressive stress test. It never chooses `K` by task or by V16 results.

We do not select [Subspace-Decomposed JEPA (SD-JEPA)](https://arxiv.org/abs/2605.31111) for V16. SD-JEPA reserves a progression subspace, samples temporal positives and negatives for a cosine-margin triplet loss, and tunes the progression dimension over `{2,4,8}` by environment. Its premise is a coherent shared notion of episode progress; the present DMC caches instead contain IID random actions and corruption intervals, so elapsed trajectory position is not a reliable task-progress label. SD-JEPA also adds a triplet margin/window/weight and optionally changes predictor conditioning and planning cost. It reports gains on three of four LeWM tasks but a regression on OGBench-Cube, with the best progression dimension varying by task. That is a worthwhile separate study, not the minimal anti-collapse intervention requested here.

## Paired-view objective

For one trajectory minibatch, let

- `Z_clean = f_theta(O_clean)` be the synchronized clean embedding sequence, encoded by the same online encoder with dropout disabled and gradients retained;
- `Z_obs = f_theta(O_train_view)` be the causally available corrupted training view; and
- `B_obs = M(Z_obs, A)` be the selected deployed memory representation (`none`, `ssm`, or `hacssmv8`).

The existing LeWorldModel predictor produces the next-clean-latent estimate from the observed path and executed action:

```text
Z_hat[t+1] = P_phi(B_obs[<=t], A[t])
L_pred = mean_t ||Z_hat[t+1] - Z_clean[t+1]||_2^2.
```

There is no stop-gradient or EMA target encoder. Rewards, simulator state, task observations, corruption masks/identities, and validation data do not enter the optimized objective.

Here, **paired-view** refers to the predictive pair: corrupted observed context predicts its synchronized active clean target. It does not mean that the synthetic corrupted embedding distribution is itself a regularization target. Applying collapse regularization to black, frozen, checkerboard, or noisy views would introduce a new invariance/distribution objective and would be less faithful to both direct Sub-JEPA and the repaired V10--V15 host.

For a SIGReg host, draw the frozen subspace matrices once from the optimizer seed. Each `P_k` has shape `d_s x D`, is initialized by QR so its rows are orthonormal, and is never optimized. Different `P_k` matrices are independent random subspaces; they are not claimed to form one mutually orthogonal partition. At every forward pass, draw a fresh set of `M=512` random unit sketch directions independently inside each subspace, matching the released Sub-JEPA implementation. The sketch directions are not learned or retained across forwards. With `R_K(Z)` denoting the mean Epps-Pulley SIGReg statistic over the fresh directions and all `K` subspaces, the objective is

```text
L_SIG,K = L_pred + 0.1 * R_K(Z_clean).
```

The regularizer acts only on the active synchronized clean target embeddings, exactly as the current repaired host applies its diversity terms to `Z_clean`. Gradients still update the shared online encoder. `Z_obs` receives gradients through `L_pred`, but no direct SIGReg/VICReg gradient.

For the VICReg control, preserve the existing V10--V15 clean-target-only variance/covariance definitions:

```text
L_VIC = L_pred + V(Z_clean) + C(Z_clean).
```

The marginal standard-deviation hinge is at one; the off-diagonal covariance square sum is divided by `D`. SIGReg has optimization weight zero in the VICReg cells. Conversely, VICReg variance and covariance have optimization weight zero in the three SIGReg host families. No cell combines both anti-collapse objectives.

## Frozen host variants

The embedding width is `D=128`, so `d_s=D/K` exactly in every SIGReg cell.

| host ID | `K` | `d_s` | role |
|---|---:|---:|---|
| `subjepa16` | 16 | 8 | primary Sub-JEPA regularizer on the active clean target |
| `subjepa32` | 32 | 4 | aggressive narrow-subspace stress test; never eligible to replace `subjepa16` post hoc |
| `fullsig` | 1 | 128 | matched full-space SIGReg control using the identical projection/statistic code path |
| `vicreg` | n/a | n/a | exact existing causal, clean-target-only VICReg host control |

`fullsig` is the matched `K=1` control, not a claim to reproduce the original LeWM checkpoint: it retains this repository's causal encoder normalization, paired views, `D=128`, current data, and current training/evaluation harness.

All SIGReg cells use:

```text
lambda = 0.1
M = 512 sketch directions per subspace
sketch direction policy = fresh unit directions every forward
projection initialization = orthogonal_frozen
projection orthogonality penalty = 0
```

The `K=1` control uses this identical fresh-direction implementation, so its comparison with `K=16` isolates the subspace construction rather than a fixed-versus-resampled sketch difference. Every checkpoint and final metric bundle must record `K`, `d_s`, `M`, `lambda`, the fresh-direction policy, projection initialization, whether projections require gradients, a digest of the frozen projection tensor, `regularizer_source=active_clean_target`, and the clean-target regularizer value. Any observed-view regularizer value may be logged only as a detached diagnostic and must have optimization weight zero.

## Frozen memory backbones and 12 design IDs

Only existing implementations are in scope:

| memory ID | exact role |
|---|---|
| `none` | the same one-token LeWorldModel predictor with the recurrent path disabled (`memory_impl=ema`, `memory_mode=none`); this is a one-frame, not three-frame, temporal baseline |
| `ssm` | existing learned diagonal-SSM memory baseline |
| `hacssmv8` | existing compact HACSSM-v8 implementation, unchanged |

The 12 training design IDs are the Cartesian product below; no additional V8 mode or V11--V15 observer is part of V16.

| host | no memory | diagonal SSM | compact V8 |
|---|---|---|---|
| primary Sub-JEPA | `subjepa16_none` | `subjepa16_ssm` | `subjepa16_hacssmv8` |
| stress Sub-JEPA | `subjepa32_none` | `subjepa32_ssm` | `subjepa32_hacssmv8` |
| matched full SIGReg | `fullsig_none` | `fullsig_ssm` | `fullsig_hacssmv8` |
| VICReg control | `vicreg_none` | `vicreg_ssm` | `vicreg_hacssmv8` |

The memory code, parameterization, initialization, recurrence, fusion, and ablation semantics must be identical across the four host variants. Host comparisons are paired within a memory backbone; memory comparisons are paired within a host.

## Exact adaptive-development grid

Tasks and fixed task/GPU assignment:

| task | GPU |
|---|---:|
| `cartpole.swingup` | 0 |
| `fish.swim` | 1 |
| `pendulum.swingup` | 2 |
| `walker.walk` | 3 |

Optimizer seeds are exactly `{16001, 16002, 16003}`. Each of the 12 designs is trained independently for every task and seed, giving 36 cells per task, 144 cells overall, and 4,320 expected epoch rows. A failed or nonfinite cell remains failed; it is never imputed, silently restarted under a new seed, or omitted from the denominator.

The data contract remains the existing V11 cache contract:

- `1,200` train episodes and `240` validation episodes;
- trajectory length `48`, RGB size `64`, train cache seed `37100`, validation cache seed `103710`;
- bounded tanh IID-Gaussian actions with `smooth_rho=0`;
- corruption seed `11012`; and
- synchronized `clean` and `train` views from the same train trajectory/action sequence.

Unless this document explicitly changes a field, training inherits the V15 host settings: 30 epochs, batch size 64, AdamW at `3e-4`, weight decay `1e-5`, gradient-norm clipping at 1, BF16 autocast, `D=128`, patch size 8, six encoder layers/four heads, four predictor layers/eight heads, history length 3, dropout 0.1, affine-free causal encoder normalization, no predictor normalization, and no scheduler, early stopping, best-checkpoint selection, or result-dependent relaunch.

Use a distinct output root and online study, proposed as:

```text
outputs/subjepa_v16_development
logs/subjepa_v16_development
W&B study: subjepa-v16-development
```

Each completed cell must contain 30 epoch rows, final checkpoint, full scalar/config metadata, state-probe payload, evaluation table, paired video, rollout NPZ, local SHA-256 digest, matching remote artifact, and a run receipt. The command manifest and all source/cache hashes must be written before launch. Artifact incompleteness makes the grid incomplete; scientific aggregates must not be produced by pretending missing cells are absent by design.

## Analysis contract

The primary outcome is the existing equal-condition held-out prior-state NMSE computed with train-only state probes and evaluation-only task observations. State/task observations remain evaluation-only and cannot select a checkpoint or update a model. Raw NMSE is compared only within the same task/seed; the cross-task summary is the equal-weight mean of paired relative effects.

Predeclared host contrasts, within each of `none`, `ssm`, and `hacssmv8`, are:

1. `subjepa16` versus `fullsig`: does moderate subspace SIGReg help beyond the matched full-space statistic?
2. `subjepa16` versus `vicreg`: does the selected Sub-JEPA host help beyond the existing host repair?
3. `subjepa32` versus `subjepa16`: what fails or improves when subspaces narrow from 8 to 4 dimensions?

The headline is `subjepa16`; `subjepa32` is a stress test, not a second candidate from which the better result may be selected. Report all task-by-seed cells, task means, equal-task paired relative effects, clean-condition harm, next-latent prediction loss, effective rank, mean channel variance, singleton/prefix causality errors, nonfinite events, and regularizer curves. Also report the host-by-memory interaction rather than attributing an effect seen under only one backbone to Sub-JEPA generally.

No threshold in this adaptive grid can produce `CONFIRMATION`, `ICLR_READY`, or a universal winner. The strongest allowed positive label is `ADAPTIVE_DEVELOPMENT_POSITIVE`; incomplete artifacts or nonfinite primary cells require an explicit incomplete/fail-closed label. There is no automatic 100-epoch continuation in this protocol.

## Scientific boundary

The four task caches, validation trajectories, corruption family, baselines, and earlier outcomes were opened repeatedly during V11--V15 development. Sub-JEPA and the choices of `K=16`, `K=32`, and these controls were selected after inspecting both those failures and May 2026 papers. Reusing the caches is useful for diagnosis and rapid comparison, but it invalidates a held-out confirmation interpretation.

Both Sub-JEPA and SD-JEPA are recent arXiv preprints rather than established results in this setting. V16 is also not a literal Sub-JEPA replication: the source uses `D=192` (`d_s=12` at `K=16`, `d_s=6` at `K=32`) and its released default uses `M=1024`, whereas this matched memory study freezes `D=128` (`d_s=8/4`) and `M=512`. The `K=1` and VICReg controls are therefore essential, and a failure of `K=32` must not be generalized to wider subspaces in the source setting.

The grid also measures representation and state-prediction behavior, not executed policy return or task success. Videos and offline rollouts are artifacts, not return evaluation. Even a complete favorable result would show only that paired-view subspace regularization helps these three existing LeWorldModel/memory backbones on this opened corruption cohort. A confirmation claim would require a frozen post-V16 method, unopened tasks and trajectories, modern tuned baselines, separately specified executed-return evaluation, multiple seeds, complete artifacts, and uncertainty computed without post-hoc architecture or hyperparameter selection.
