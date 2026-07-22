# Event-Versioned CEM on Official Frozen DINO-WM

## Verdict

- Truly official DINO-WM: `true`.
- Proxy conclusions transfer: `false`.
- Task / seeds: `official_dinowm_wall` / `[0, 1, 2]`.
- Cue window supplied to model: `false`.

## Exact model artifact

`{'adapter': 'scripts.run_dinowm_wall_stage_h.FrozenWallHost', 'cache_manifest': 'outputs/dinowm_wall_audit_v1/stage_h_carriers/cache/manifest.json', 'cache_manifest_sha256': 'b9dffde1a43763e46548b69d751bcaa04bbd472368af91d7af9b9e7cf90b73b4', 'checkpoint': 'outputs/dinowm_wall_audit_v1/checkpoint/model_latest.pth', 'checkpoint_epoch': 65, 'checkpoint_sha256': '8441971becdae934fe08de5b163398390a32f6fe1fb0a8df113290e2468e142b', 'digest_after': 'bf4e325cf0fdba2f6376451e4b364151833883840054a88b11fa7124339514e6', 'digest_before': 'bf4e325cf0fdba2f6376451e4b364151833883840054a88b11fa7124339514e6', 'source': 'outputs/dinowm_native_pusht_audit_v1/vendor/dino_wm', 'source_revision': '0a9492fa12044b852ae9e001cc74604b79c8bb0c', 'unchanged': True, 'dinov2_source': 'outputs/dinowm_native_pusht_audit_v1/vendor/dinov2', 'dinov2_source_revision': '7764ea0f912e53c92e82eb78a2a1631e92725fc8', 'dinov2_encoder_weights': 'outputs/dinowm_native_pusht_audit_v1/torch_home/hub/checkpoints/dinov2_vits14_pretrain.pth', 'dinov2_encoder_weights_sha256': 'b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'}`

## Exact aggregate results

### immediate_overwrite

- Overwrite / false-write: 0.2194 / 0.0000
- Full / reset / no-state BAcc: 0.4000 / 0.2431 / 0.2431
- Host loss full / reset: 1.311476 / 1.396614
- Version selection / stale / fallback: 0.2194 / 0.0000 / 0.0000
- Retrieval precision / recall: 0.2194 / 0.2194
- High-CE / random deletion Δloss: 0.085139 / 0.000000
- Occupancy / evictions: 1.0000 / 0.00

### hysteresis_only

- Overwrite / false-write: 0.7507 / 0.0003
- Full / reset / no-state BAcc: 0.7736 / 0.2444 / 0.2444
- Host loss full / reset: 1.159860 / 1.396614
- Version selection / stale / fallback: 0.7507 / 0.0333 / 0.0000
- Retrieval precision / recall: 0.7507 / 0.7507
- High-CE / random deletion Δloss: 0.236754 / 0.000000
- Occupancy / evictions: 1.0000 / 0.00

### version_store_no_verification

- Overwrite / false-write: 0.5514 / 0.1993
- Full / reset / no-state BAcc: 0.6347 / 0.2361 / 0.2361
- Host loss full / reset: 1.222819 / 1.396614
- Version selection / stale / fallback: 0.5514 / 0.0035 / 0.0000
- Retrieval precision / recall: 0.5514 / 0.5514
- High-CE / random deletion Δloss: 0.173795 / 0.000000
- Occupancy / evictions: 3.0000 / 1358.67

### full_versioned_delayed_verification

- Overwrite / false-write: 0.7507 / 0.0000
- Full / reset / no-state BAcc: 0.7583 / 0.2542 / 0.2542
- Host loss full / reset: 1.164679 / 1.396614
- Version selection / stale / fallback: 0.7507 / 0.0333 / 0.0000
- Retrieval precision / recall: 0.9574 / 0.7507
- High-CE / random deletion Δloss: 0.295813 / 0.000000
- Occupancy / evictions: 1.8785 / 126.67

## Proxy comparison

- Proxy: `{'overwrite_correctness': 0.8174545159194282, 'false_write_rate': 0.11879528605665982, 'full_bacc': 0.7762369406264531, 'source': 'outputs/cem_event_versioning_v1/report.json'}`
- Official conclusion transfer: `false`.

## Scope and caveats

The checkpoint, predictor, action/proprio encoders and DINOv2 features are official and frozen. Overwrite timing is constructed by relocating cached genuine cue tokens in latent time rather than rerendering and re-encoding every randomized schedule.
- Event discovery uses the official frozen predictor's one-step latent surprise. Ground-truth event timing is evaluator-only.
- The controller and post-hoc audit readouts are outside the frozen host. Host parameters are hashed before and after every seed.
- This is a real official Wall checkpoint evaluation, but not a claim about native DINO-WM planning.

## Artifacts

- `outputs/cem_event_versioning_dinowm_official_v1/report.json`
- `outputs/cem_event_versioning_dinowm_official_v1/wall/s<seed>/{result.json,decision_log.json}`
- `docs/assets/cem_event_versioning_dinowm_official_factorial.{png,pdf}`
- `docs/assets/cem_event_versioning_dinowm_official_lifelines.{png,pdf}`

## Execution

- Completed cells: 3.
- Device policy: `cuda:1`; `cuda:3` is rejected.
- Jobs still running: [].
