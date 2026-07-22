# CEM Automatic Cue Discovery V2

## Verdict

**Automatic discovery usable:** `false`.
The model received no cue window, cue onset, duration, or readout time; ground-truth event intervals were attached only after inference.

## Exact aggregate outcomes

### cube-single-play-v0

- Seeds: [0, 1, 2]

**v1_immediate_surprise**
- Boundary F1 / IoU: 0.6271 / 0.7271
- Cue write recall / distractor false-write: 1.0000 / 0.9598
- Promotion precision: 0.1467
- Retrieval precision / recall: 0.4084 / 0.4132
- Overwrite correctness: 0.3217
- Full / reset / no-state BAcc: 0.5366 / 0.2579 / 0.2579
- Host loss with / without memory: 1.6681 / 1.4420
- Mean occupancy / write budget: 6.8524 / 1.0000
- Deletion CE, selected-high / random: 0.9875 / 0.6367

**provisional_grouping**
- Boundary F1 / IoU: 0.6271 / 0.7271
- Cue write recall / distractor false-write: 1.0000 / 0.9511
- Promotion precision: 0.3630
- Retrieval precision / recall: 0.8228 / 0.8346
- Overwrite correctness: 0.7061
- Full / reset / no-state BAcc: 0.7375 / 0.2579 / 0.2579
- Host loss with / without memory: 0.8419 / 1.4420
- Mean occupancy / write budget: 2.7048 / 0.3947
- Deletion CE, selected-high / random: 0.9875 / 0.6367

**delayed_ce_verification**
- Boundary F1 / IoU: 0.6271 / 0.7271
- Cue write recall / distractor false-write: 0.8301 / 0.1317
- Promotion precision: 0.6981
- Retrieval precision / recall: 0.8788 / 0.7765
- Overwrite correctness: 0.7199
- Full / reset / no-state BAcc: 0.7486 / 0.2579 / 0.2579
- Host loss with / without memory: 0.8187 / 1.4420
- Mean occupancy / write budget: 1.1667 / 0.1703
- Deletion CE, selected-high / random: 0.9875 / 0.6367

**full_v2_hysteresis_router**
- Boundary F1 / IoU: 0.6271 / 0.7271
- Cue write recall / distractor false-write: 0.8301 / 0.1317
- Promotion precision: 0.6981
- Retrieval precision / recall: 0.8788 / 0.7765
- Overwrite correctness: 0.7199
- Full / reset / no-state BAcc: 0.7486 / 0.2579 / 0.2579
- Host loss with / without memory: 0.8187 / 1.4420
- Mean occupancy / write budget: 1.1667 / 0.1703
- Deletion CE, selected-high / random: 0.9875 / 0.6367

- Best thresholds by seed: [{'delta': 0.0, 'margin': 0.0, 'tau': 2}, {'delta': -0.05, 'margin': 0.0, 'tau': 2}, {'delta': 0.1, 'margin': 0.0, 'tau': 2}]

### pointmaze-large-navigate-v0

- Seeds: [0, 1, 2]

**v1_immediate_surprise**
- Boundary F1 / IoU: 0.6749 / 0.7793
- Cue write recall / distractor false-write: 1.0000 / 0.9769
- Promotion precision: 0.1679
- Retrieval precision / recall: 0.2524 / 0.2611
- Overwrite correctness: 0.3078
- Full / reset / no-state BAcc: 0.3739 / 0.2579 / 0.2579
- Host loss with / without memory: 2.2135 / 1.4374
- Mean occupancy / write budget: 5.7571 / 1.0000
- Deletion CE, selected-high / random: 0.9049 / 0.5086

**provisional_grouping**
- Boundary F1 / IoU: 0.6749 / 0.7793
- Cue write recall / distractor false-write: 1.0000 / 0.9741
- Promotion precision: 0.4052
- Retrieval precision / recall: 0.8476 / 0.8861
- Overwrite correctness: 0.7368
- Full / reset / no-state BAcc: 0.7335 / 0.2579 / 0.2579
- Host loss with / without memory: 0.8807 / 1.4374
- Mean occupancy / write budget: 2.3619 / 0.4103
- Deletion CE, selected-high / random: 0.9049 / 0.5086

**delayed_ce_verification**
- Boundary F1 / IoU: 0.6749 / 0.7793
- Cue write recall / distractor false-write: 0.8212 / 0.1116
- Promotion precision: 0.7377
- Retrieval precision / recall: 0.8877 / 0.7865
- Overwrite correctness: 0.7244
- Full / reset / no-state BAcc: 0.7579 / 0.2579 / 0.2579
- Host loss with / without memory: 0.8370 / 1.4374
- Mean occupancy / write budget: 1.0667 / 0.1853
- Deletion CE, selected-high / random: 0.9049 / 0.5086

**full_v2_hysteresis_router**
- Boundary F1 / IoU: 0.6749 / 0.7793
- Cue write recall / distractor false-write: 0.8212 / 0.1116
- Promotion precision: 0.7377
- Retrieval precision / recall: 0.8877 / 0.7865
- Overwrite correctness: 0.7244
- Full / reset / no-state BAcc: 0.7579 / 0.2579 / 0.2579
- Host loss with / without memory: 0.8370 / 1.4374
- Mean occupancy / write budget: 1.0667 / 0.1853
- Deletion CE, selected-high / random: 0.9049 / 0.5086

- Best thresholds by seed: [{'delta': 0.02, 'margin': 0.0, 'selection': 'lexicographic target-count then utility on held-out sweep', 'tau': 2}, {'delta': 0.0, 'margin': 0.0, 'selection': 'lexicographic target-count then utility on held-out sweep', 'tau': 2}, {'delta': 0.0, 'margin': 0.0, 'selection': 'lexicographic target-count then utility on held-out sweep', 'tau': 2}]

## Success targets

- false_write_below_0_25: `true`
- overwrite_above_0_75: `false`
- full_bacc_at_least_0_75: `true`
- controls_at_most_0_35: `true`
- host_loss_not_worsened: `true`

## Design and failure modes

- Surprise pulses enter a short provisional buffer and adjacent onset/offset pulses are grouped before any persistent write.
- A learned CE estimator is calibrated against true task-loss change under group deletion. Promotion requires two of three delayed estimates to exceed delta.
- Capacity replacement requires CE improvement by the hysteresis margin; query routing sees verified events only.
- If all targets do not hold simultaneously, the Pareto figure and every delta/tau/margin point are retained in each cell result.
- Remaining failures are quantified by the failed target flags above, rather than inferred from boundary quality alone.
- Verification fixes the v1 false-write and host-loss failures, but automatic discovery is not fully usable under the requested conjunction: mean overwrite correctness is 0.7199 (cube) and 0.7244 (pointmaze), below the strict >0.75 target. The retained Pareto sweeps show that admitting lower-CE groups raises cue recall but does not remove this routing conflict consistently across seeds.

## Artifacts

- `outputs/cem_auto_discovery_v2/report.json`
- `outputs/cem_auto_discovery_v2/<env>/s<seed>/{result.json,decision_log.json,model.pt}`
- `docs/assets/cem_auto_discovery_v2_timeline.svg`
- `docs/assets/cem_auto_discovery_v2_factorial.svg`
- `docs/assets/cem_auto_discovery_v2_pareto.svg`

## Execution

- Completed cells: 6.
- GPU smoke test: passed before the focused sweep.
- Focused sweep: 2 environments × 3 seeds × 384 episodes, with disjoint train/tuning/held-out splits.
- Device policy: `cuda:2`; the runner rejects `cuda:3`.
- Jobs still running: [].
