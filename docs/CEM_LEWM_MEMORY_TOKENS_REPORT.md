# CEM LeWM Dedicated Memory Tokens Report

## Mechanism

The official predictor is fixed to 3 positions. Six distinct memory slots are keys/values for cross-attention queried by the complete frozen predictor's 3-token output; the gated adapter output plus an order-preserving linear map of all six token positions is merged at that output boundary without slot pooling.

The released predictor cannot directly accept appended tokens: its `pos_embedding` has shape `(1, 3, 192)`. The implemented equivalent is complete frozen predictor (including `pred_proj`) → cross-attention from its three output-boundary tokens to six distinct memory tokens → gated merge. No memory mean-pooling occurs.

## Results

- Seeds: [0, 1, 2]; age: 15.
- Trainable parameters: 1,306,184.
- Host-output BAcc mean: 0.8194.
- Host loss with memory mean: 0.131307; without memory: 0.009664.
- Host-loss increase from exposure: 0.121643 (13.59× the frozen-host loss).
- All success criteria passed: True.
- Frozen digest unchanged for every seed: True.
- Labels used in training loss: false.

Per-seed ladder and all controls are recorded in `report.json`; the figure plots every seed and the identical v3 dense-residual baseline.

## Causal deletion and overhead

- Cue-group deletion Δloss mean: 0.137333.
- Matched-random deletion Δloss mean: -0.001767.
- Measured predictor overhead mean: 18.72%.

The six-way counterfactual candidates are rendered variants of the same base trajectory. Semantic class labels are used only after training for fail-closed linear-probe diagnostics.

Dedicated tokens therefore pass the requested identity-exposure and control gates, but this is not an overall host-loss win: strict next-latent predictive fidelity degrades substantially.
