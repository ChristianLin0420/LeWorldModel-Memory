# EVENT-VERSIONED CEM Discovery V3

## Verdict

- Overwrite target >0.80 passed: `true`.
- All requested quantitative gates passed: `true`.
- No cue window, onset, duration, or query delay was supplied to the store or router.

## Host-scope qualification

Frozen v2 OGBench event-host proxy; official DINO-WM weights were not loaded by this runner. Each trained event host was frozen before the factorial/sweep, and its parameter digest was asserted unchanged. Consequently these runs are real cached OGBench visual-task evaluations, but they are not evidence about unchanged official DINO-WM weights.

## Exact aggregate results

### cube-single-play-v0

- Seeds: [0, 1, 2]
- Selected thresholds: [{'delta': 0.05, 'margin': 0.0, 'persistence_steps': 1, 'selection': 'held-out lexicographic gate-count then overwrite-weighted utility'}, {'delta': 0.0, 'margin': 0.0, 'persistence_steps': 2, 'selection': 'held-out lexicographic gate-count then overwrite-weighted utility'}, {'delta': 0.1, 'margin': 0.0, 'persistence_steps': 1, 'selection': 'held-out lexicographic gate-count then overwrite-weighted utility'}]

**immediate_overwrite_v1**
- Overwrite / false-write: 0.7055 / 0.7457
- Full / control BAcc: 0.6736 / 0.2590
- Host loss with / without memory: 1.1582 / 1.4842
- Version selection / stale error / fallback: 0.7186 / 0.3327 / 0.0000
- Retrieval precision / recall: 0.7082 / 0.7186
- Mean occupancy / capacity evictions: 2.0952 / 2.33
- High-CE / random deletion Δloss: 0.9235 / 0.6094

**hysteresis_only**
- Overwrite / false-write: 0.6391 / 0.6509
- Full / control BAcc: 0.6885 / 0.2590
- Host loss with / without memory: 1.1998 / 1.4842
- Version selection / stale error / fallback: 0.8008 / 0.4829 / 0.0000
- Retrieval precision / recall: 0.7895 / 0.8008
- Mean occupancy / capacity evictions: 2.0952 / 2.33
- High-CE / random deletion Δloss: 0.9235 / 0.6094

**version_store_no_verification**
- Overwrite / false-write: 0.8193 / 0.8252
- Full / control BAcc: 0.5215 / 0.2590
- Host loss with / without memory: 1.2069 / 1.4842
- Version selection / stale error / fallback: 0.5683 / 0.1030 / 0.0000
- Retrieval precision / recall: 0.5600 / 0.5683
- Mean occupancy / capacity evictions: 2.4619 / 17.00
- High-CE / random deletion Δloss: 0.9235 / 0.6094

**full_versioned_delayed_verification**
- Overwrite / false-write: 0.8193 / 0.1289
- Full / control BAcc: 0.7722 / 0.2590
- Host loss with / without memory: 0.7433 / 1.4842
- Version selection / stale error / fallback: 0.7814 / 0.2538 / 0.0057
- Retrieval precision / recall: 0.8949 / 0.7814
- Mean occupancy / capacity evictions: 1.1524 / 0.00
- High-CE / random deletion Δloss: 0.9235 / 0.6094

### pointmaze-large-navigate-v0

- Seeds: [0, 1, 2]
- Selected thresholds: [{'delta': 0.05, 'margin': 0.0, 'persistence_steps': 1, 'selection': 'held-out lexicographic gate-count then overwrite-weighted utility'}, {'delta': 0.05, 'margin': 0.0, 'persistence_steps': 1, 'selection': 'held-out lexicographic gate-count then overwrite-weighted utility'}, {'delta': 0.0, 'margin': 0.0, 'persistence_steps': 1, 'selection': 'held-out lexicographic gate-count then overwrite-weighted utility'}]

**immediate_overwrite_v1**
- Overwrite / false-write: 0.6565 / 0.7943
- Full / control BAcc: 0.6580 / 0.2590
- Host loss with / without memory: 1.1720 / 1.4810
- Version selection / stale error / fallback: 0.7762 / 0.4064 / 0.0000
- Retrieval precision / recall: 0.7429 / 0.7762
- Mean occupancy / capacity evictions: 1.9952 / 1.67
- High-CE / random deletion Δloss: 0.9590 / 0.5042

**hysteresis_only**
- Overwrite / false-write: 0.6390 / 0.7397
- Full / control BAcc: 0.6917 / 0.2590
- Host loss with / without memory: 1.1597 / 1.4810
- Version selection / stale error / fallback: 0.8409 / 0.4640 / 0.0000
- Retrieval precision / recall: 0.8048 / 0.8409
- Mean occupancy / capacity evictions: 1.9952 / 1.67
- High-CE / random deletion Δloss: 0.9590 / 0.5042

**version_store_no_verification**
- Overwrite / false-write: 0.8249 / 0.9310
- Full / control BAcc: 0.5360 / 0.2590
- Host loss with / without memory: 1.0970 / 1.4810
- Version selection / stale error / fallback: 0.6070 / 0.0716 / 0.0000
- Retrieval precision / recall: 0.5810 / 0.6070
- Mean occupancy / capacity evictions: 2.2905 / 5.00
- High-CE / random deletion Δloss: 0.9590 / 0.5042

**full_versioned_delayed_verification**
- Overwrite / false-write: 0.8156 / 0.1087
- Full / control BAcc: 0.7803 / 0.2590
- Host loss with / without memory: 0.7811 / 1.4810
- Version selection / stale error / fallback: 0.8114 / 0.2523 / 0.0000
- Retrieval precision / recall: 0.9214 / 0.8114
- Mean occupancy / capacity evictions: 1.0571 / 0.00
- High-CE / random deletion Δloss: 0.9590 / 0.5042

## Success gates

- overwrite_above_0_80: `true`
- false_write_below_0_20: `true`
- full_bacc_at_least_0_75: `true`
- controls_at_most_0_35: `true`
- host_loss_nonworsening: `true`
- high_ce_deletion_above_random: `true`

## Architecture and failures

- Every discovered semantic key owns timestamped versions with value, CE estimate, confidence, and lifecycle status.
- Full v3 keeps the old verified version while a candidate remains provisional. Promotion requires persistent delayed CE improvement over the active same-key version by the selected hysteresis margin.
- Retrieval ranks all live versions by verified CE, semantic confidence, and a recency kernel, then verifies candidates in order. Failed newest candidates fall back to an older verified version.
- Eviction occurs only under total version-budget pressure and removes the lowest-CE live version.
- Any failed gate above is retained as an explicit failure; the sweep and decision logs are sufficient to inspect rejected, superseded, selected, fallback, and capacity-evicted versions.

## Artifacts

- `outputs/cem_event_versioning_v1/report.json`
- `outputs/cem_event_versioning_v1/<env>/s<seed>/{result.json,decision_log.json,model.pt}`
- `docs/assets/cem_event_versioning_lifelines.{png,pdf}`
- `docs/assets/cem_event_versioning_factorial.{png,pdf}`
- `docs/assets/cem_event_versioning_pareto.{png,pdf}`
- `docs/assets/cem_event_versioning_capacity_delay.{png,pdf}`

## Execution

- Completed cells: 6.
- Requested device policy: `cuda:1`; the runner rejects `cuda:3`.
- Jobs still running: [].
