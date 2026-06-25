# LeWorldModel (LeWM) + Two-Timescale Memory

PyTorch implementation of "LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels" (arXiv:2603.19312), **extended with a two-timescale (short/long-term) memory** for a foundational study of how memory shapes the dynamics of a JEPA latent space (targeting ICLR 2027).

> **Memory study TL;DR.** JEPA encoders are memoryless and the LeWM predictor only sees a 3-frame window. We add two exponential-moving-average memory banks over the latent stream — a *fast* (short-term, `τ≈3`) and a *slow* (long-term, `τ≈25`) leaky integrator — injected into the predictor with zero-init projections (so training starts exactly at the baseline). The loss stays 2-term. We then *visualize how short- vs long-term memory affect the decision* across 4 memory-stressing environments. See **[`docs/PROPOSAL.md`](docs/PROPOSAL.md)** (method + math + experiments) and **[`docs/RESEARCH_BRIEF.md`](docs/RESEARCH_BRIEF.md)** (annotated literature review).

### Memory study quickstart
```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt wandb pyyaml   # torch: use cu128 wheels for Blackwell
.venv/bin/wandb login                                  # logs to wandb project "lewm-memory"
.venv/bin/python scripts/test_memory.py                # unit tests for the EMA math + model
EPOCHS=30 NUM_EPISODES=5000 bash scripts/run_all.sh    # 4 GPUs: one memory env each x {none,short,long,both}
.venv/bin/python scripts/aggregate_results.py          # env x design summary table + figure
```

Key new files: `lewm/models/memory.py` (two-timescale EMA + fusion), `lewm/models/memory_model.py` (`MemoryLeWorldModel`), `lewm/envs/memory_envs.py` (tmaze/occlusion/recall/distractor + tworoom control), `lewm/eval/memory_probe.py` (availability + usage probes), `scripts/train_memory.py`, `scripts/run_all.sh`.

---

## Base LeWM

## Architecture

- **Encoder**: ViT-Tiny (patch_size=14, 12 layers, 3 heads, embed_dim=192) with [CLS] token + MLP projector + BatchNorm
- **Predictor**: 6-layer transformer with 16 attention heads, AdaLN action conditioning, 10% dropout
- **Regularizer**: SIGReg (Sketched-Isotropic-Gaussian Regularizer) - enforces Gaussian-distributed latents via random projections + Epps-Pulley normality test
- **Training Loss**: L = L_pred + λ * SIGReg(Z) — only 2 terms, 1 hyperparameter (λ=0.1)
- **Planning**: CEM (Cross-Entropy Method) in latent space with MPC

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
│       └── probing.py          # Latent probing, VoE, planning evaluation
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

## Key Paper Details

- ~15M parameters total
- Trains on single GPU in a few hours
- 48x faster planning than foundation-model-based WMs
- No EMA, stop-gradient, frozen encoders, or reconstruction loss
- Only 1 tunable hyperparameter (λ)
