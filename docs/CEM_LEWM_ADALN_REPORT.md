# CEM LeWM AdaLN Memory Report

## Result

PASS: prediction-level conditioning crosses the frozen-host exposure boundary with fail-closed controls. QUALIFICATION: it does not preserve the frozen host's future-latent loss.

Three seeds were run on six-way PushT visual binding at evidence age 15.
Memory-only, conditioning-vector, and frozen-host-output balanced accuracies
were **0.9611 ± 0.0100**, **0.8542 ± 0.0230**,
and **0.8132 ± 0.0391**, respectively. The required host-output
gate is 0.75. Controls were: reset 0.1667, no_state 0.1715, host_only 0.1715, shuffled 0.1653, random 0.1549; the ceiling is 0.217.

## Architecture and frozen-host contract

The official LeWM API exposes `action_encoder(actions)` before the predictor.
Its output is the per-token conditioning tensor consumed by all six
`ConditionalBlock.adaLN_modulation` paths. The adapter writes six distinct CEM
slots, retrieves them with context/action/time queries, and adds the resulting
192-D vector to the action embedding. Context latents are never perturbed.

Only the adapter and surprise-gate temperature train. Frozen host digest:
`5589632959b98370ad96001523025bc265686e82b87376d327da18cbd555f879`; unchanged across every seed:
**True**.

## Objectives and causal checks

Training uses the frozen host's next-latent MSE over the legal three-frame
window and six same-base rendered counterfactual branches. The observed branch
is selected by latent equality, not a semantic class id. Labels are opened only
for the post-hoc ridge-probe audit.

Deleting the old cue group yields host-output BAcc
**0.1563 ± 0.0029**, versus **0.8132 ± 0.0380**
for count-matched random deletion. Cue deletion was more causal in
3/3
seeds. Mean conditioning norm was **39.7039 ± 3.6494**.

## Host loss and overhead

Future-latent loss with/without memory was
**0.3009 ± 0.0090 / 0.0097 ± 0.0000**.
Thus the stated exposure/control gate passes, but this is not a loss-preserving
integration: future loss is **31.1x** baseline. The adapter repurposes frozen
prediction capacity to expose the old cue rather than improving the host's
ordinary next-latent prediction.
The adapter has 943,497 trainable
parameters versus 18,034,478 frozen host
parameters (5.232%).
Measured batch-latency overhead was
1699.7% on the
assigned GPU (includes prefix surprise and retrieval, batch size 128).

## Dense-residual v3 comparison and limits

The prior v3-D dense residual path reached host-output BAcc
0.1556. Architecture C instead
uses the predictor's real AdaLN conditioning interface, so this is not a
surrogate hook. It is still an additive action-embedding intervention rather
than a new per-block side input; the same memory vector reaches each block only
through the frozen block-specific AdaLN projections.

Artifacts: `outputs/cem_lewm_adaln_memory_v1/report.json` and
`docs/assets/cem_lewm_adaln_exposure.{png,pdf}`.
