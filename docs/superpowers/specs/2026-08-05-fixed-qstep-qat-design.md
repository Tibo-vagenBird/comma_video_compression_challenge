# Fixed-Q-Step QAT for Cool-Chic Residue Synthesis

Date: 2026-08-05

## Objective

Add a conservative quantization-aware training (QAT) stage to the modified
Cool-Chic encoder used by `experiments/coolchic_baseline`. The stage should
recover part of the measured post-training quantization tax without increasing
the encoded synthesis-network rate.

The first version deliberately targets only the residue Cool-Chic synthesis
weights and biases. It does not combine QAT with the planned Muon optimizer
experiment, change the bitstream format, learn quantization steps, or alter the
decoder.

## Motivation and Evidence

The quantization-tax probe found that residue synthesis dominates the observed
quality loss:

- Frame 1: residue synthesis segmentation tax `+0.00060527`
- Frame 3: residue synthesis segmentation tax `+0.00042216`
- ARM and inter-frame modules: no measured segmentation tax
- Residue upsampling: much smaller tax
- Motion synthesis: negligible tax

QAT is therefore a targeted attempt to make residue synthesis parameters robust
to the exact quantization used in the submitted bitstream. It is not expected to
solve Cool-Chic's larger per-frame decoder-rate disadvantage by itself.

## Selected Strategy

Use calibration followed by fixed-step QAT:

1. Complete normal full-precision Cool-Chic training.
2. Save the full-precision residue synthesis weights and biases.
3. Run the existing `quantize_model()` once. It performs the existing
   entropy-aware discrete RD search and records the selected synthesis weight
   and bias steps in `coolchic_enc["residue"].nn_q_step.synthesis`.
4. Keep those selected steps but restore the saved full-precision residue
   synthesis parameters as trainable master parameters.
5. Freeze every parameter except residue synthesis weights and biases.
6. Train with fixed-step fake quantization for at most 1,000 optimizer steps.
7. Select a byte-valid QAT checkpoint, hard-quantize it with the fixed selected
   steps, and run the existing RDOQ once.

The q-steps remain fixed throughout QAT. The master parameters remain floating
point and may move between quantization bins. Consequently, the transmitted
integer values and their entropy-coded size may still change.

## Why the QAT Stage Precedes RDOQ

RDOQ is a final discrete search over already-quantized parameter values. QAT is
a gradient-based local adaptation of floating-point master parameters. Training
after RDOQ would overwrite or invalidate RDOQ's discrete decisions.

The required order is therefore:

```text
float training
  -> save float residue-synthesis parameters
  -> quantize_model() q-step search
  -> restore float residue-synthesis masters
  -> fixed-q-step QAT
  -> hard quantization with the saved q-steps
  -> RDOQ
  -> bitstream encoding
```

The hard-quantization step after QAT must use the saved steps directly. It must
not rerun the q-step search.

## Fake-Quantization Semantics

For a trainable master tensor `x` and fixed scalar q-step `q`, the forward pass
uses:

```text
x_fake = x + stop_gradient(round(x / q) * q - x)
```

The forward value is exactly `round(x / q) * q`, matching Cool-Chic's hard
quantizer. The backward derivative with respect to `x` is the identity
straight-through estimator. Cool-Chic's integer representation is not an INT8
format, so this operation must not add an INT8 clamp.

Weights and biases use their separately selected fixed q-steps.

The implementation must not swap tensors through `.data`, as that can silently
break autograd. The preferred integration is an opt-in fake-quantization path in
`SynthesisConv2d.forward()`. It is disabled by default and activated only on the
residue synthesis instance during QAT. The transient QAT state is not serialized
into a model or bitstream.

## Trainable and Frozen State

Trainable during QAT:

- `frame_encoder.coolchic_enc["residue"].synthesis` weights
- `frame_encoder.coolchic_enc["residue"].synthesis` biases

Frozen during QAT:

- All residue latent grids
- Residue ARM, upsampling, and inter-frame modules
- The residue output transform
- The complete motion Cool-Chic
- All quantization steps
- Reference frames and target data

Stored Exp-Golomb counts are not gradient-trained. Candidate measurement uses
the production network encoder, including its existing count re-selection for
delta-coded networks. For non-delta roots, the count selected by
`quantize_model()` remains in force until RDOQ.

The QAT setup must explicitly set `requires_grad` from an allowlist rather than
assuming the preceding training phase left the correct flags in place. It must
assert that at least one target weight and one target bias are trainable, and
that no non-target parameter is trainable.

## Optimization and Loss

QAT v1 uses SOAP for the trainable residue synthesis parameter group. Muon is a
separate experiment so the QAT-on/off comparison changes only one lever.

Defaults:

```text
COMMA_QAT=0
COMMA_QAT_ITERS=1000
COMMA_QAT_LR=2e-5
COMMA_QAT_PATIENCE=300
COMMA_QAT_FREQ_VALID=50
COMMA_QAT_FREQ_BYTES=100
```

`COMMA_QAT=0` is the compatibility default and must retain the original
quantize-then-RDOQ path.

The QAT objective reuses the existing challenge-tuned training loss and the
current frame's training phase lambda and distortion weights. No new
differentiable network-rate proxy is added in v1. Because latents and all
non-target modules are frozen, the QAT stage is a local reconstruction and
challenge-distortion repair; exact network rate is enforced during checkpoint
selection.

Use a maximum of 1,000 optimizer steps, require at least 200 steps before
patience-based termination, validate every 50 steps, and stop after 300 steps
without a better valid checkpoint. A short test or smoke run may override the
iteration variables.

## Checkpoint and Rate Policy

Before the first update, evaluate step 0 using the restored master parameters
under fake/hard quantization and record:

- Validation loss and its distortion components
- Exact production-encoded residue Cool-Chic network payload bytes
- The fixed q-steps and Exp-Golomb counts
- A copy of the full-precision master weights and biases

Every validation interval, hard-quantize a non-destructive candidate copy with
the fixed steps and compute its metrics. Every byte interval, call the production
`encode_network()` path and measure the full residue Cool-Chic network payload.
ARM, upsampling, and inter-frame parameters are frozen, so changes relative to
step 0 come from synthesis, while measuring the full payload avoids inventing a
module-only serialization format. Candidate evaluation must not mutate the
trainable master model.

For a delta-coded P-frame, byte measurement must receive the exact selected
trained reference Cool-Chic used by the real encoder. It must not reconstruct a
reference from a newly initialized model or silently measure the candidate as a
full network. Step 0 and every candidate use the same reference object.

A candidate is eligible only if its exact production network payload bytes do
not exceed the step-0 value. Among eligible candidates, retain the one with the
best existing RD objective, evaluated with the exact payload bits and the frozen
latent rate. Step 0 is always a valid fallback. If no later candidate improves
it, restore step 0 before final hard quantization and RDOQ.

The per-frame byte gate cannot guarantee the final ZIP size. ZIP compression
depends on the complete bitstream, including later frames. Final acceptance is
therefore based on two complete experiments:

- `QAT_OFF`: control archive
- `QAT_ON`: QAT candidate archive

The existing temporary-ZIP scorer compares their actual archive sizes and
challenge scores. The QAT result is retained only if the complete scored result
is better under the experiment's rate constraint.

## Integration Boundaries

The feature should be split into small responsibilities:

1. **Fixed-step fake quantizer**: a pure autograd-safe tensor operation.
2. **Synthesis opt-in path**: uses fake-quantized weights and biases only while
   QAT is active.
3. **QAT controller**: validates configuration, snapshots/restores state,
   freezes parameters, runs optimization, evaluates candidates, and returns the
   selected floating-point master state.
4. **Quantization pipeline hook**: invokes QAT between `quantize_model()` and
   `rdoq_model()` only when enabled.
5. **Experiment plumbing**: prints resolved defaults and passes environment
   configuration through the existing cloud-run workflow.

The decoder and bitstream syntax remain unchanged because the final model is
hard-quantized and encoded by the existing production path.

## Logging

At QAT startup, log:

- Enabled state and resolved configuration
- Frame identity
- Number and names of trainable tensors
- Selected weight and bias q-steps
- Step-0 validation metrics and exact network bytes

At validation, log:

- Step
- Training and validation loss
- Segmentation distortion, MSE, and pose metric when available
- Exact bytes when measured
- Whether the checkpoint is byte-valid and whether it became the best

At completion, log the selected step, stopping reason, metric change from step
0, byte change from step 0, and whether fallback was used.

## Error Handling and Safety

QAT must fail safely for the current frame. If any of the following occurs,
restore step 0, disable fake quantization, and continue through the original
hard-quantization/RDOQ path:

- Missing or non-positive synthesis q-step
- Missing synthesis weight or bias parameters
- Empty trainable parameter set
- A non-target parameter remains trainable
- Non-finite training loss or gradients
- Candidate hard quantization fails
- Exact byte measurement fails

Exceptions that indicate programming errors should still include a precise
diagnostic in the log. Cleanup must run through `try/finally` so transient fake
quantization state is never left enabled.

## Tests

### Unit tests

- Fixed-step fake quantization produces values identical to the existing hard
  quantizer for positive, negative, zero, and half-step inputs.
- The straight-through estimator gives gradients to the floating-point master
  tensor.
- Weight and bias can use different q-steps.
- Enabling and disabling synthesis fake quantization does not alter stored
  parameters.
- QAT setup marks only residue synthesis weights and biases trainable.
- Saved q-steps remain unchanged across optimization steps.
- Hard candidate evaluation does not mutate master parameters.
- Invalid configuration and non-finite state select step 0 safely.

### Compatibility tests

- With `COMMA_QAT=0`, the original quantization/RDOQ call order and output are
  unchanged.
- Existing bitstream-version, header-version, frame-reconstruction, delta, and
  archive-rate tests continue to pass.

### Integration tests

- A small iteration count completes the full sequence:
  q-step search -> QAT -> fixed-step hard quantization -> RDOQ -> encode.
- The resulting bitstream decodes successfully and incremental reconstruction
  remains exact.
- A QAT-on/off pilot reports exact temporary-ZIP sizes and challenge metrics for
  both complete runs.

## Pilot and Success Criteria

Run a controlled comparison with identical seed, source frame data, training
preset, and all non-QAT environment values:

```text
QAT_OFF
QAT_500
QAT_1000
```

Primary success criteria:

- Recover at least 30% of the measured residue synthesis quantization tax on
  the probe frames.
- Do not increase the final actual archive size relative to the matched control.
- Do not introduce pose instability or decode mismatch.

If 500 and 1,000 steps are equivalent, adopt 500 as the future default. If the
metric is still improving at 1,000 steps without rate growth, a later experiment
may test 2,000. Dynamic, learnable, or alternating q-step strategies are out of
scope until fixed-step QAT demonstrates a useful signal.

## Non-Goals

- Muon integration
- QAT for motion, ARM, upsampling, inter-frame modules, latents, or output
  transforms
- Learned or dynamically updated q-steps
- New bitstream fields or format version
- A differentiable Exp-Golomb rate proxy
- Automatic branching of two complete video encodes inside one run
