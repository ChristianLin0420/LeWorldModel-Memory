# CEM on Raw OGBench Renderings

## Result

Causal-Effect Memory (CEM) was evaluated on 18 cells covering 10 OGBench
environments and four families. Every cell used original cached renderings,
actions, and timestamps. The raw protocol improved future-latent prediction
relative to the same frozen host without memory in all 10 environments.
The mean cell-level reduction was **0.719%** (95% CI **0.621%--0.818%**), and
the gain increased from **0.227%** at rollout step 1 to **1.086%** at step 4.

This is a modest predictive result, not a solved event-discovery result. A
recent-only state remained slightly stronger on average
(`0.48320` versus CEM `0.48383` MSE), and CEM beat recent-only in only 3 of 18
cells. Predicted causal-effect ranking was weak (mean held-out Spearman
`0.037`). High-ranked event deletion exceeded matched random deletion in 12 of
18 cells, but the aggregate gap's confidence interval included zero. The raw
result supports useful persistent state beyond the host window; it does not
show that CEM consistently improves on a simple recent-state extension.

## Strict no-manual-event contract

The runner enforces and records the following contract in every feature
receipt, cell result, decision log, and aggregate report.

- Consumed cache arrays: `frames` and `actions` only. Timestamps are array
  indices.
- Cached `cue_labels`, `cue_positions`, and all other metadata arrays are
  listed as ignored and are never read by model or training code.
- `cue_window` is `null`; `cue_window_used_by_model` is `false`.
- No call is made to `draw_cue`, `inject_cue_sequence`,
  `inject_cue_sequence_mode`, or `_saliency_map`. An AST audit checks this at
  runtime.
- No manually selected frame, event label, fixed event interval, color rule,
  saturation rule, saliency map, reward, goal, or simulator state supplies a
  write or keep target.
- WRITE proposals use frozen DINOv2 patch-token change and frozen-host
  prediction surprise, with thresholds estimated from train trajectories.
- KEEP targets are future action-conditioned latent-loss reductions under
  singleton event-group keep/delete comparisons. RECALL uses learned
  query/content/age/need scores. A bounded true-loss verification audit is
  computed after the future target is observed and is never used to select the
  primary prediction.
- The architecture receipt selects its displayed event by the largest proposal
  score among retrieved test events. Its four frame hashes are computed from
  the unmodified cache bytes.

The machine-readable contract is under
`outputs/cem_raw_ogbench/report.json::no_manual_cue_contract`. All 18 cells
passed it.

## Protocol

### Data and split

Each cache contains 768 trajectories of 22 frames. A fixed, recorded
trajectory-level permutation produces 70% train, 15% validation, and 15% test
splits (538/115/115 trajectories). PCA, host training, CEM training, threshold
selection, and final evaluation respect this split:

1. Frozen DINOv2 ViT-S/14 produces `x_norm_patchtokens`.
2. A 1x1 plus 2x2 spatial pyramid is formed from the 14x14 patch grid.
3. A 96-dimensional PCA projection is fit on train trajectories only.
4. A finite-window, action-conditioned latent predictor is trained on train
   trajectories, selected on validation loss, then frozen and hashed.
5. Host surprise and DINO semantic changes propose adjacent event groups.
6. CEM trains a memory residual, delayed group-effect estimator, semantic
   version keys, hysteresis store, and bounded router on train trajectories.
7. Promotion threshold, delay, budget, hysteresis, and top-k are selected on
   validation trajectories. Test trajectories are reported once.

The breadth host is a **DINO-feature action-conditioned world model**, also
described as a **DINO-WM-style host**. It is not the released DINO-WM model.

### Coverage and seeds

PointMaze-large, PointMaze-teleport, Cube-single, and Cube-triple use three
optimization seeds each. PointMaze-giant, Cube-double, Puzzle-3x3, Scene,
AntMaze-large, and HumanoidMaze-large use one breadth seed each. The
trajectory split is fixed across optimization seeds and its hashes are
recorded. No environment was excluded: the AntMaze and HumanoidMaze hosts
passed the preregistered finite/reliable check.

## Host predictor quality

Every host beat both a train-mean baseline and latent persistence on test.
`host/persist` is one-step host MSE divided by persistence MSE.

| environment | seeds | host MSE | host/persist | reliable |
|---|---:|---:|---:|---:|
| PointMaze-large | 3 | 0.2761 | 0.866 | 3/3 |
| PointMaze-teleport | 3 | 0.3365 | 0.858 | 3/3 |
| PointMaze-giant | 1 | 0.2687 | 0.848 | 1/1 |
| Cube-single | 3 | 0.3806 | 0.856 | 3/3 |
| Cube-double | 1 | 0.3741 | 0.862 | 1/1 |
| Cube-triple | 3 | 0.3519 | 0.872 | 3/3 |
| Puzzle-3x3 | 1 | 0.5694 | 0.789 | 1/1 |
| Scene | 1 | 0.4544 | 0.828 | 1/1 |
| AntMaze-large | 1 | 0.2082 | 0.873 | 1/1 |
| HumanoidMaze-large | 1 | 0.1627 | 0.883 | 1/1 |

Host digests are identical before and after CEM training in every cell.

## Test results

`memory/no memory` is future-latent MSE averaged over a four-step rollout.
Deletion values are the increase in loss after deleting the indicated
singleton automatic event group. `rho` is held-out Spearman correlation
between predicted and measured group effects.

| environment | n | memory/no memory | reduction | high/random deletion | rho |
|---|---:|---:|---:|---:|---:|
| AntMaze-large | 1 | 0.3026/0.3057 | 1.02% | 0.0027/0.0035 | -0.005 |
| Cube-double | 1 | 0.5269/0.5308 | 0.74% | 0.0048/0.0048 | 0.053 |
| Cube-single | 3 | 0.5331/0.5373 | 0.79% | 0.0050/0.0050 | 0.109 |
| Cube-triple | 3 | 0.4974/0.5021 | 0.94% | 0.0054/0.0051 | 0.030 |
| HumanoidMaze-large | 1 | 0.2389/0.2415 | 1.07% | 0.0042/0.0046 | -0.053 |
| PointMaze-giant | 1 | 0.3874/0.3895 | 0.53% | 0.0020/0.0019 | 0.135 |
| PointMaze-large | 3 | 0.4188/0.4208 | 0.49% | 0.0030/0.0028 | 0.045 |
| PointMaze-teleport | 3 | 0.5201/0.5236 | 0.67% | 0.0049/0.0044 | 0.010 |
| Puzzle-3x3 | 1 | 0.7382/0.7418 | 0.49% | 0.0042/0.0041 | 0.008 |
| Scene | 1 | 0.6069/0.6096 | 0.43% | 0.0037/0.0038 | -0.050 |

Family-level prediction reductions were 0.846% for manipulation, 0.678% for
navigation, 0.494% for puzzle, and 0.433% for scene.

### Controls and telemetry

Across the 18 cells, mean future-latent MSE was:

| condition | MSE |
|---|---:|
| CEM | 0.48383 |
| no memory | 0.48725 |
| reset memory | 0.48725 |
| shuffled episode memory | 0.48939 |
| random matched-norm memory | 0.48952 |
| recent-only state | 0.48320 |

The mean proposal count was 1.923 events per query. Of delayed verification
decisions, 58.6% were promoted and 41.4% rejected. Mean occupancy was 0.901,
and 80.3% of queries retrieved at least one event. Relative to eligible
decisions, same-key fallback occurred in 17.7%, supersession in 8.5%, and
capacity eviction in 3.2%; 9.9% of retrieved selections used an older active
version. The checked-in budget and delay curves show the full sweeps rather
than a hand-picked age.

The separate retrieve-then-verify audit accepts 64.6% of routed candidates
(95% CI 62.9%--66.2%) when acceptance requires measured singleton group
future-loss reduction above zero. Its post-hoc verified-memory MSE is 0.48137.
This audit uses the observed future target and is reported separately; it does
not alter the leakage-free primary prediction or its metrics.

### Causal deletion and calibration

Mean high-ranked and random deletion effects were `0.004253` and `0.004147`.
The high-minus-random gap was `0.000106` with 12/18 cell wins; its 95% CI
`[-0.000056, 0.000269]` includes zero. Mean CE-hat Spearman was `0.037`
(`0.010` to `0.065` across cells), positive in 13/18 cells but too small for a
strong calibration claim. Scene has a negative family-level deletion gap.

### Downstream action use

The evaluated downstream endpoint is **post-hoc action-sequence
identification**, not planning. Each query ranks the observed four-action
sequence against seven episode-shuffled sequences by latent rollout error;
retrieval is selected without seeing candidate actions. Mean top-1 accuracy is
`0.3066` with memory and `0.3030` without memory, with 8/18 cell wins. Existing
native controllers were not used because they consume synthetic goal labels,
not the raw memory representation; adapting them would violate the protocol.

## Official DINO-WM result is separate

The existing Wall result remains an exact, separate protocol using the
released frozen DINO-WM checkpoint and its cached patch-token bank. Its CEM
condition reports full/reset BAcc `0.758/0.254`, future-latent loss
`1.165/1.397`, and high/random deletion `0.296/0.000` over three seeds. It
relocates cached encoded events and is not raw OGBench breadth. None of the
breadth environments above is labeled official DINO-WM.

## Negative results and limitations

1. CEM does not beat recent-only on average and wins that comparison in only
   3/18 cells. The raw result therefore does not establish better event
   selection than a one-step context extension.
2. CE-hat calibration is weak, and high-versus-random deletion is not
   significant at the aggregate level. Host-loss reduction is more consistent
   than causal ordering.
3. Four-step, 22-frame cached trajectories do not test unbounded memory.
4. Breadth uncertainty is asymmetric: four representative environments have
   three seeds; six breadth environments have one.
5. Renderings are simulated but unmodified. They are not natural images.
6. Action-sequence identification is an offline use test, not executed
   planning or control.
7. PCA and the memory adapter are environment-specific. Cross-environment
   transfer was not evaluated.

## Reproduction

```bash
# Contract and end-to-end smoke.
.venv/bin/python -m pytest -q scripts/test_run_cem_raw_ogbench.py
.venv/bin/python scripts/launch_cem_raw_ogbench.py --smoke --gpus 1 2

# Full breadth campaign. The launcher rejects GPU3.
.venv/bin/python scripts/launch_cem_raw_ogbench.py \
  --campaign --gpus 0 1 2

# Exact figures, table fragments, and paper snapshot.
.venv/bin/python scripts/plot_cem_raw_ogbench.py
```

## Artifacts

- Aggregate: `outputs/cem_raw_ogbench/report.json`
- Launch receipt: `outputs/cem_raw_ogbench/launch_receipt.json`
- Cells: `outputs/cem_raw_ogbench/cells/<env>/s<seed>/`
- Per-cell files: `result.json`, `decision_log.json`, `model.pt`
- Raw-frame receipt: `outputs/cem_raw_ogbench/figure_receipt.json`
- Figures: `docs/assets/cem_raw_{event_timeline,rollout_horizon,causal_deletion,budget_pareto,family_aggregate}.{pdf,png}`
- Paper snapshot: `paper_d/generated_results/raw_ogbench_snapshot.json`
- Paper tables: `paper_d/generated_results/raw_ogbench_{main,hosts}.tex`

No jobs remain running, no cell failed after retries, and GPU3 was not used.
