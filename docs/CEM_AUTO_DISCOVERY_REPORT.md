# CEM Automatic Cue-Time and Readout Discovery

## Scope

CEM receives rendered frames, actions, normalized time, and a visual decision-context query. It never receives cue onset, duration, or a cue window. Ground-truth intervals are joined only after inference.

**Verdict:** automatic discovery works = `false`.
Boundary discovery and memory readout work partially, but the full criterion fails because overwrite routing is unreliable and memory does not reduce mean host loss.

## Exact aggregate results

### cube-single-play-v0

- Seeds: [0, 1, 2]
- WRITE F1 / IoU: 0.6447 / 0.7273
- Distractor false-write rate: 0.9196
- Retrieval precision / recall: 0.6176 / 0.6333
- Overwrite correctness: 0.3436
- Full / reset / no-state BAcc: 0.5899 / 0.2687 / 0.2644
- Host loss with / without memory: 1.4391 / 1.3867

### pointmaze-large-navigate-v0

- Seeds: [0, 1, 2]
- WRITE F1 / IoU: 0.6695 / 0.7704
- Distractor false-write rate: 0.9670
- Retrieval precision / recall: 0.6373 / 0.6974
- Overwrite correctness: 0.3308
- Full / reset / no-state BAcc: 0.5811 / 0.2735 / 0.2644
- Host loss with / without memory: 1.4639 / 1.3858

## Failure cases and interpretation

- A surprise gate intentionally writes matched-salience distractors; query-time routing, not WRITE, must reject them.
- Mean host loss is higher with memory than without it in both environments despite improved balanced accuracy; the readout is poorly calibrated.
- Latest-cue overwrite correctness remains below 0.50 in both environments, so automatic temporal routing is not solved.
- Adjacent events closer than the maximum injected duration can merge into one discovered interval.
- Very slow background motion or a one-frame low-contrast event can fall below the episode-calibrated surprise threshold.

## Artifacts

- `outputs/cem_auto_discovery_v1/report.json`
- `outputs/cem_auto_discovery_v1/<env>/s<seed>/decision_log.json`
- `docs/assets/cem_auto_discovery_timeline.{png,pdf}`
- `docs/assets/cem_auto_discovery_metrics.{png,pdf}`

## Testing

- GPU smoke test: passed on `cuda:0`.
- Real focused grid: 2 OGBench environments × 3 seeds, 240 episodes per cell, completed.
- Python compilation and linter checks: passed.

No jobs were left running when this report was generated.
