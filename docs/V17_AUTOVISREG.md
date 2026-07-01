# V17: coefficient-free Wasserstein–VISReg development study

## Scope

V17 is a 72-cell excluded adaptive-development study on the already-opened V11
trajectory caches. It is not confirmation evidence and it does not evaluate
executed control return. The candidate changes only the clean-target
anti-collapse update in the repaired causal LeWorldModel host. The encoder,
one-token action-conditioned predictor, optimizer, paired corruption path,
`none`/SSM/HACSSMv8 memories, probes, held-out corruptions, and rollout contract
remain unchanged.

No teacher, target stop-gradient, projection MLP, prototype bank, negative
samples, labels, rewards, new memory architecture, or observation-correction
branch is introduced.

## What V16 actually failed on

V16 completed all 144 cells, but the Sub-JEPA candidate did not beat VICReg and
its apparent advantage over full-space SIGReg came from a blind frame rather
than improved geometry.

1. The active clean target and observed path share the encoder, so predictive
   MSE admits a constant-code solution.
2. At an empirical delta distribution, the Epps–Pulley SIGReg gradient is
   common-mode; at projected value zero it is exactly zero. It therefore stops
   supplying sample-diversifying force just when it is needed.
3. V16's independently initialized subspace blocks form a badly conditioned
   stacked frame even though their row count sums to `D`. Median condition
   numbers were about 522 for K=16 and 682 for K=32, versus 1 for the K=1
   orthogonal control. Trapped learned codes had frame Rayleigh quotients around
   `1.7e-4` and `2.5e-4`.
4. The scalar loss was not too small. Trapped K=16 cells averaged raw SIGReg
   `25.7465`, weighted regularization `2.57465`, prediction `0.000207`, and a
   regularizer/prediction ratio above 12,000. This is the analytic projected-zero
   plateau (`25.731` at batch 64), not Gaussian matching.
5. Collapse was task-dependent and bistable: pendulum trapped in all 18 K=16/32
   cells, walker in none, with late stochastic escapes around epochs 7–29.
   Even escaped Sub-JEPA representations stayed around effective rank 3. VICReg
   averaged rank 32.2.

The V17 requirement is therefore not a larger SIGReg coefficient, more
subspaces, more epochs, or another memory. It is a full-space diversity geometry
that directly sees missing eigen-directions and retains a recovery gradient,
plus a loss combiner that does not expose a task-selected coefficient.

## Online method screen

- [VISReg](https://arxiv.org/abs/2606.02572) was designed specifically for JEPA
  collapse. It identifies SIGReg's vanishing collapse gradient and separates
  center, scale, and sliced-Wasserstein shape. The released method still uses a
  recipe-dependent outer lambda and a selectable slice count, so V17 is not an
  official reproduction.
- [Rethinking the Uniformity Metric in Self-Supervised Learning](https://proceedings.iclr.cc/paper_files/paper/2024/file/21bcef9a879b85714387f94d7ecc2c91-Paper-Conference.pdf)
  derives a closed-form Gaussian quadratic-Wasserstein uniformity metric and
  shows that it detects feature redundancy and dimensional collapse. Its
  covariance-spectrum term is the narrow repair V16 needs.
- [VICReg](https://arxiv.org/abs/2105.04906) remains the positive control because
  its separate variance and covariance terms were the only V16 family with
  consistent diversity gradients and useful rank.
- Gradient normalization motivates balancing by actual shared-parameter
  gradients, but [GradNorm](https://proceedings.mlr.press/v80/chen18a.html)
  retains an asymmetry hyperparameter. V17 instead uses the unique angular
  bisector of the two normalized encoder gradients and restores the original
  prediction-gradient norm.

## Candidate objective

Let clean active-target embeddings be pooled as
`Z in R^(N x D)`, where `N=B*T`, and let `mu` and `Sigma` be their biased-N
empirical mean and covariance. The full-spectrum term is squared Gaussian W2 to
the isotropic target, normalized by dimension:

```text
L_W2 = mean(mu^2) + mean_i (sqrt(lambda_i(Sigma) + 1e-6) - 1)^2.
```

The affine-free host LayerNorm fixes each token norm near `sqrt(D)`, so `I` is
the corresponding raw-coordinate isotropic target. Unlike coordinate variance
plus off-diagonal covariance, this is one rotation-invariant distance: a missing
eigendirection and a wrong scale are penalized by the same geometry, with no
relative coefficient.

VISReg supplies the exact-collapse tie breaker. Standardize each coordinate by
its detached empirical standard deviation, take `K=2D` fresh normalized
full-space Gaussian slices, sort each projection, and match standard-normal
quantiles. Its self-paced multiplier is

```text
g = clamp(mean coordinate standard deviation, 0, 1)
L_reg = L_W2 + g * L_shape.
```

`g` is measured in units of the objective's own target scale. It is neither an
epoch schedule nor a selected threshold. It cancels VISReg's `1/std` shape
amplification near LeWorldModel's tied code and reaches full strength at the
isotropic target. The structural slice rule `K=2D` is derived from embedding
dimension and is not exposed on the CLI.

For the shared encoder, let `p` and `r` be the prediction and regularizer
gradients. Positive rescaling of either scalar loss must not alter the update.
For nonzero, non-antiparallel gradients V17 uses

```text
u = p / ||p|| + r / ||r||
g_encoder = ||p|| * u / ||u||.
```

The bisector has positive dot product with both raw gradients and its norm is
exactly the original prediction-gradient norm. Predictor and memory parameters
receive their prediction-only gradients. Exact zero and antiparallel cases have
explicit finite fallbacks; the inherited global norm clip remains unchanged.

Thus there is no selectable V17 SSL coefficient, temperature, momentum,
prototype count, slice count, gate schedule, or task/memory-specific setting.
The existing optimizer learning rate, weight decay, batch size, epoch budget,
and host clip remain training-protocol settings and are not claimed to vanish.

## Excluded numerical preflight

These checks selected the mechanism before the 72-cell protocol was frozen and
are not grid evidence.

- Direct VISReg stayed at variance around `1e-6` because its shape gradient
  dominated the scale direction.
- Gating shape restored variance to about 0.98 by epoch 10, but rank plateaued
  near 3.
- Standardized covariance created a low-scale rank-one trap (variance 0.0068,
  rank 1.02 at epoch 15).
- Raw covariance improved only to rank 6.89 at epoch 15; gating it reached rank
  4.01 at epoch 12.
- Separately normalizing tiny covariance gradients produced numerical rank by
  decorrelating microscopic noise while variance remained around `3e-5`; it was
  rejected.
- The final W2-spectrum plus gated-shape candidate crossed rank 16 at epoch 7
  on the cartpole/no-memory stress cell and reached rank 29.6 with mean channel
  variance 0.34 at epoch 12 while predictive loss continued downward.

No rejected arm is part of the frozen grid.

## Frozen grid and gates

The grid is:

```text
families:  autovisreg, vicreg
memories:  none, ssm, hacssmv8
tasks:     cartpole.swingup, fish.swim, pendulum.swingup, walker.walk
seeds:     17001, 17002, 17003
epochs:    30
cells:     2 * 3 * 4 * 3 = 72
```

The `autovisreg` label denotes the full coefficient-free W2+VISReg candidate
above. `vicreg` is an exact same-seed V16 host control. Both no-memory arms run
first on each task-pinned GPU.

The analyzer reports paired candidate-minus-control effects for held-out prior
state NMSE, clean prior NMSE, and predictive loss; channel variance and
covariance trace before rank; effective rank with a validity flag; late-window
prediction/regularizer convergence; actual encoder gradient norms and cosine;
adaptive scale; conflicts; pre-clip norm and clip fraction; and artifact/W&B
integrity. Rank recovery is predeclared as effective rank at least 16 with
nontrivial covariance trace. Scientific failures do not invalidate otherwise
complete adaptive-development artifacts.

Every online cell must contain `model.pt`, `metrics.json`, `eval_rollout.npz`,
and a finished `wandb_run.json` receipt with the exact entity, project, study,
run URL/ID, artifact name, and rollout SHA. The runner refuses mixed source,
data, command, or protocol hashes on resume.

## Commands

```bash
.venv/bin/python scripts/test_train_autovisreg_v17.py
.venv/bin/python scripts/test_autovisreg_v17_pipeline.py
.venv/bin/python scripts/run_autovisreg_v17.py --dry-run

PYTHONUNBUFFERED=1 nohup .venv/bin/python scripts/run_autovisreg_v17.py \
  > logs/autovisreg_v17_runner.log 2>&1 < /dev/null &
```

The outer runner log is deliberately outside the runner's fresh per-cell log
directory. Resume only after the previous process is dead, with the identical
command plus `--resume`.
