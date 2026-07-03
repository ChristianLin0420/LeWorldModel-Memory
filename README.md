# Persistent state in a finite-context LeWorldModel-derived JEPA

This repository contains a PyTorch implementation of *LeWorldModel: Stable
End-to-End Joint-Embedding Predictive Architecture from Pixels*
([arXiv:2603.19312](https://arxiv.org/abs/2603.19312)) and a sequence of studies
of explicit persistent state in pixel JEPAs.

Published LeWorldModel (LeWM) is not memoryless or noncausal: its predictor is
action-conditioned, temporally causal, and attends to a configured finite
observation history (`H=3` for PushT and OGBench-Cube and `H=1` for TwoRoom).
The narrower architectural limitation is that it has no explicit recurrent or
belief state that survives after an observation leaves that finite window.

V18 tests whether compact SAS-PC/V8 supplies useful persistent state under
partial observability. All eight V18 designs share a causally normalized,
active-clean-target, **VICReg-trained LeWM-derived architecture host**. This is
not the exact SIGReg-based LeWM method, and the result must not be summarized as
“V8 improves LeWorldModel.” V18 evaluates prior-state prediction only: it does
not execute a policy and makes no return, success, control, or planning claim.
See the [frozen V18 protocol](docs/V18_LEWM_V8_CONFIRMATION.md) and the
[complete architecture/evidence record](docs/LEARNABLE_MEMORY.md).

## V18 confirmation result

> **Analysis-rendered receipt.** The release renderer fills every value below
> from the validated write-once analysis/CSV bundle and computes displayed file
> hashes from those exact bytes. The registered decision is conjunctive; a
> favorable subset cannot change it. The renderer fails on any leftover token.

The frozen study contains five previously unopened DMC tasks, eight designs,
five optimizer seeds, and 100 epochs: 200 cells in total. Its primary endpoint
is equal-task, equal-seed held-out prior task-state NMSE under four registered
corruption families.

| receipt | final analyzer value |
|---|---|
| analyzer status | `COMPLETE` |
| scientific label | `CONFIRMATION_FAILED` |
| official confirmation result | `false` |
| completed valid cells | `200/200` |
| artifact integrity | `PASS` |

| registered comparison or guard | mean paired reduction / observation | 95% CI | cell wins | task wins | gate |
|---|---:|---:|---:|---:|---:|
| SAS-PC vs per-cell better GRU/SSM | `-2.10%` | `[-13.35%, +9.11%]` | `11/25` | `2/5` | `FAIL` |
| SAS-PC vs no persistent carrier | `+12.44%` | `[+3.65%, +22.43%]` | `24/25` | `5/5` | `PASS` |
| SAS-PC vs legal initial-frame/action integrator | `-38.12%` | `[-61.38%, -20.58%]` | `0/25` | `0/5` | `FAIL` |
| deep-gap persistence vs selected GRU/SSM | `-1.74%` | `[-12.95%, +9.46%]` | `11/25` | `2/5` | `FAIL` |
| recurrent action-transport intervention | `+9.48%` | `[+2.88%, +16.09%]` | `23/25` | `5/5` | `PASS` |
| joint two-state-read intervention | `-5.39%` | `[-13.12%, +0.67%]` | `6/25` | `1/5` | `FAIL` |
| learned shrinkage vs static/dynamic envelope | `-1.89%` | `[-4.88%, +0.37%]` | `6/25` | `0/5` | `FAIL` |
| clean-prior guard vs selected GRU/SSM | `+11.73%` | `[+5.82%, +17.70%]` | `25/25` | `5/5` | `PASS` |
| representation health | min variance `0.0228`; min rank `2.02` | — | variance `200/200`; rank `144/200` | — | `FAIL` |
| convergence | max absolute late change `+132.09%` | — | `126/200` | — | `FAIL` |

**Registered interpretation:** CONFIRMATION_FAILED: at least one registered conjunctive clause fails, so SAS-PC/V8 is not confirmed as a generally superior persistent carrier; favorable individual comparisons remain descriptive and cannot rescue it.

Two process interruptions required five complete-cell replacements: four SSM
cells (`acrobot.swingup`, `manipulator.bring_ball`, `quadruped.run`, and
`swimmer.swimmer15`, all seed 18005) after the first interruption, and the
`stacker.stack_4` static-SAS-PC cell at seed 18003 after the second. All five
interrupted cells lacked core artifacts; their replacements restarted at epoch
one and reached epoch 100. The schema-v2 review receipt binds the exact cells,
logs, terminal W&B states, and final result hashes.

The frozen protocol registered V8 gate/route telemetry, but the exporter
retained only final shrinkage coefficients and action-feature norms. Per-step
gate vectors and route weights are unavailable. This secondary-report deviation
does not enter or affect any primary preregistered gate.

Regardless of label, this is evidence about persistent information transport
and named interventions in the stabilized VICReg host. It is not evidence of
improved executed return or planning, does not establish causal discovery or
causal representations, and does not test the exact original SIGReg objective.

Review-safe copies of the canonical result artifacts:

- [`confirmation_analysis.json`](paper/review_artifact/confirmation_analysis.json) — write-once decision; SHA-256 `f471076e63aa8f9b0c4e4f666459b3abf58cbc71d934ea147937195f72e04c58`.
- [`confirmation_cells.csv`](paper/review_artifact/confirmation_cells.csv) — all 200 cell receipts; SHA-256 `71144b471acbf88ae1f67c91615299effc7f6f53cc97d578e0b9da3cf8021c23`.
- [`confirmation_contrasts.csv`](paper/review_artifact/confirmation_contrasts.csv) — registered contrasts; SHA-256 `78b804c3acfc433c4276a8b6dbab999486573a55d7c005581cff91206f800d92`.

## V18 execution and result rebuild

The V18 implementation is split into the
[cohort adapter](scripts/hacssm_v18_data.py),
[trainer](scripts/train_lewm_v8_v18.py),
[four-GPU runner](scripts/run_lewm_v8_v18.py), and
[write-once analyzer](scripts/analyze_lewm_v8_v18.py). The runner requires a
clean committed frozen source tree, the complete cohort manifest, task-pinned
GPUs `0,1,2,3`, and finished online W&B receipts.

```bash
# Focused frozen-contract checks (31 tests).
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/pytest -p no:cacheprovider -q \
  scripts/test_train_lewm_v8_v18.py \
  scripts/test_run_lewm_v8_v18.py \
  scripts/test_analyze_lewm_v8_v18.py

# Collect or validate the complete frozen cohort and its manifest.
.venv/bin/python scripts/hacssm_v18_data.py --all

# Inspect the frozen 200-cell expansion without training.
.venv/bin/python scripts/run_lewm_v8_v18.py --dry-run

# Use exactly one of these: fresh launch, or cell-granular resume.
.venv/bin/python scripts/run_lewm_v8_v18.py
.venv/bin/python scripts/run_lewm_v8_v18.py --resume

# Revalidate from a later release tree without rewriting the frozen decision.
# `none` checks local caches, checkpoints, histories, artifacts, and all gates;
# use `full` to additionally query all 200 remote runs and download rollouts.
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  scripts/audit_lewm_v8_v18_final.py \
  --repo "$PWD" \
  --root "$PWD/outputs/lewm_v8_v18_confirmation" \
  --wandb-check none > /tmp/v18_final_audit.json
```

The runner invokes the analyzer with `--write` exactly once after all 200 cells
validate. The official analyzer is also bound to the frozen execution commit's
clean Git receipt, so invoking it from a later release commit correctly refuses;
the independent auditor above verifies the frozen commit bytes through Git
without changing the write-once result. Authors retaining the private
raw-artifact tree can regenerate the
provenance-bound figures with:

```bash
.venv/bin/python scripts/plot_v18_paper.py \
  --root outputs/lewm_v8_v18_confirmation \
  --output-dir docs/figures
```

To regenerate the checked-in manuscript, figures, and review artifact from the
private frozen outputs, use the repository paths explicitly:

```bash
.venv/bin/python scripts/plot_v18_paper.py \
  --root outputs/lewm_v8_v18_confirmation \
  --output-dir docs/figures

.venv/bin/python scripts/render_v18_paper.py \
  --root outputs/lewm_v8_v18_confirmation \
  --log-root logs/lewm_v8_v18_confirmation \
  --restart-audit docs/V18_RESTART_AUDIT.json \
  --template templates/ICLR.template.md \
  --output docs/ICLR.md \
  --manifest-output docs/ICLR.manifest.json

REVIEW_STAGE="$(mktemp -d)"
.venv/bin/python scripts/build_v18_review_artifact.py \
  --root outputs/lewm_v8_v18_confirmation \
  --log-root logs/lewm_v8_v18_confirmation \
  --protocol-document docs/V18_LEWM_V8_CONFIRMATION.md \
  --restart-audit docs/V18_RESTART_AUDIT.json \
  --output "$REVIEW_STAGE/review_artifact"

# Existing checked-in bundles are write-protected. Compare the staged bundle;
# replace paper/review_artifact only after all validation and review checks pass.
diff -ru paper/review_artifact "$REVIEW_STAGE/review_artifact"
```

The checked-in `docs/ICLR.md`, its manifest, `docs/figures`, and
`paper/review_artifact` are sufficient for a clean-clone paper rebuild:

```bash
PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH" \
  .venv/bin/python paper/build_paper.py \
    --source docs/ICLR.md \
    --paper-dir paper \
    --review-artifact paper/review_artifact \
    --pandoc /tmp/pandoc-3.10/bin/pandoc

PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH" \
  .venv/bin/python paper/check_v18_paper.py --compile \
    --paper-dir paper \
    --manuscript docs/ICLR.md \
    --review-artifact paper/review_artifact \
    --output paper/paper_check.json
```

The manuscript release is intentionally bound to the registered falsification
label and its provenance checks fail closed on result, figure, or source drift.
See the [paper README](paper/README.md) for the template-version caveat and
toolchain details.

## Historical fixed-EMA memory study

This earlier exploratory study predates the adaptive V8 sequence and the
prospectively frozen V18 confirmation. Its two hand-set EMA banks and synthetic
environments are historical baselines, not the V18 method or evidence.

### Historical fixed-EMA quickstart

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt wandb pyyaml   # torch: use cu128 wheels for Blackwell
.venv/bin/wandb login                                  # logs to wandb project "lewm-memory"
.venv/bin/python scripts/test_memory.py                # unit tests for the EMA math + model
EPOCHS=30 NUM_EPISODES=5000 bash scripts/run_all.sh    # 4 GPUs: one memory env each x {none,short,long,both}
.venv/bin/python scripts/aggregate_results.py          # env x design summary table + figure
```

Historical EMA files: `lewm/models/memory.py` (two-timescale EMA + fusion),
`lewm/models/memory_model.py` (`MemoryLeWorldModel`),
`lewm/envs/memory_envs.py` (tmaze/occlusion/recall/distractor + TwoRoom
control), `lewm/eval/memory_probe.py` (availability + usage probes),
`scripts/train_memory.py`, and `scripts/run_all.sh`.

---

## Base LeWM

## Architecture

- **Encoder**: ViT-Tiny (patch_size=14, 12 layers, 3 heads, embed_dim=192) with [CLS] token + MLP projector + BatchNorm
- **Predictor**: 6-layer transformer with 16 attention heads, AdaLN action conditioning, 10% dropout
- **Regularizer**: SIGReg (Sketched-Isotropic-Gaussian Regularizer) - enforces Gaussian-distributed latents via random projections + Epps-Pulley normality test
- **Training Loss**: L = L_pred + λ * SIGReg(Z) — only 2 terms, 1 hyperparameter (λ=0.1)
- **Base-paper planning evaluation**: CEM (Cross-Entropy Method) in latent space with MPC

## Project Structure

```
LeWorldModel/
├── lewm/
│   ├── models/
│   │   ├── encoder.py          # ViT-Tiny encoder + Predictor (AdaLN transformer)
│   │   ├── sigreg.py           # SIGReg regularizer
│   │   └── leworldmodel.py     # Full model combining all components
│   ├── envs/
│   │   └── two_room.py         # TwoRoom navigation environment
│   └── eval/
│       └── probing.py          # Latent probing, VoE, base-LeWM planning evaluation
├── scripts/
│   ├── train.py                # Training script
│   └── test_model.py           # Unit tests
├── configs/
│   └── default.yaml            # Default configuration
└── requirements.txt
```

## Quick Test

```bash
python scripts/test_model.py
```

## Training

```bash
# With synthetic data (for testing)
python scripts/train.py --use-synthetic --epochs 10 --batch-size 64

# With real data
python scripts/train.py --data-path /path/to/trajectories.npz --epochs 10
```

## Key Base-Paper Details

- ~15M parameters total
- Trains on single GPU in a few hours
- The base LeWM paper reports 48x faster planning than foundation-model-based WMs
- No EMA, stop-gradient, frozen encoders, or reconstruction loss in the base LeWM method
- Only 1 tunable hyperparameter (λ) in the base LeWM objective
