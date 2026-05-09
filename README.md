# LeWorldModel (LeWM) Implementation

PyTorch implementation of "LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels" (arXiv:2603.19312).

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
