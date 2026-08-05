# Fixed-Q-Step Cool-Chic QAT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, fixed-q-step QAT stage that trains only residue synthesis weights and biases between Cool-Chic's existing q-step search and final RDOQ.

**Architecture:** Reuse the full-precision snapshot already stored by `FrameEncoder._store_full_precision_param()`, while retaining the synthesis q-steps selected by `quantize_model()`. An autograd-safe synthesis-layer fake quantizer supplies hard-quantized forward values with straight-through gradients; a focused controller owns freezing, SOAP optimization, exact production-network byte measurement, checkpoint selection, restoration, and final hard quantization. The existing trained same-role delta reference is loaded once by `cc_encode.py` and passed to both QAT measurement and final encoding.

**Tech Stack:** Python 3.11, PyTorch 2.11+, Cool-Chic's `SOAP`, `unittest`, Cool-Chic Exp-Golomb/delta network coder, Bash experiment launcher.

## Global Constraints

- Preserve current code style and public call patterns; read each touched function before editing.
- Do not revert or accidentally commit existing uncommitted changes in either repository.
- `COMMA_QAT=0` is the default and must retain the current quantize-then-RDOQ behavior.
- Select q-steps once with the existing `quantize_model()` RD search; never update or re-search them during QAT.
- Train only residue synthesis `main_branch` and `stabiliser_branch` weights and biases; exclude `output_transform`.
- Freeze latent grids, residue non-synthesis modules, the complete motion encoder, q-steps, and reference data.
- Defaults are exactly: 1,000 maximum steps, learning rate `2e-5`, patience 300, validation every 50 steps, byte measurement every 100 steps, and minimum 200 steps before patience can terminate training.
- Use SOAP for QAT v1; Muon remains a separate experiment.
- Use no INT8 clamp, no learnable q-step, no differentiable network-rate proxy, and no bitstream-format change.
- Candidate bytes must come from production `encode_network()` using the exact trained same-role reference when delta coding is enabled.
- Restore step 0 on failure and always clear transient fake-quant state and restore original `requires_grad` flags.
- Judge final success with separate complete QAT-off and QAT-on temporary-ZIP scores.
- Because `D:\myCodePycharm\Cool-Chic` is already dirty, inspect `git diff` and `git diff --cached` before every commit; use `git add -p` for any already-modified file such as `cc_encode.py`.

## File Structure

### Cool-Chic repository

- Modify `D:\myCodePycharm\Cool-Chic\coolchic\component\core\synthesis.py`: fixed-step STE operation and opt-in per-layer QAT state.
- Create `D:\myCodePycharm\Cool-Chic\coolchic\nnquant\qat.py`: environment configuration, target-state helpers, exact candidate evaluation, QAT loop, fallback, and final hard quantization.
- Modify `D:\myCodePycharm\Cool-Chic\coolchic\component\video.py`: call QAT between `quantize_model()` and `rdoq_model()`.
- Modify `D:\myCodePycharm\Cool-Chic\cc_encode.py`: load one exact same-role reference and reuse it for QAT and final encoding.
- Create `D:\myCodePycharm\Cool-Chic\test\test_qat_synthesis.py`: fake-quant and synthesis-state unit tests.
- Create `D:\myCodePycharm\Cool-Chic\test\test_qat.py`: configuration, state isolation, exact-reference measurement, fallback, and controller tests.
- Create `D:\myCodePycharm\Cool-Chic\test\test_qat_reference.py`: exact-reference loader/reuse tests.

### Challenge repository

- Modify `experiments/coolchic_baseline/encode_video.py`: include resolved QAT settings in the active-lever banner.
- Modify `experiments/coolchic_baseline/run_one.sh`: document QAT environment variables and record them in the concise config line.
- Modify `experiments/coolchic_baseline/README.md`: document QAT ordering, defaults, fresh-run requirement, and controlled pilot commands.
- Create `experiments/coolchic_baseline/test_qat_config.py`: banner/default regression tests.

---

### Task 1: Autograd-Safe Fixed-Step Fake Quantization

**Files:**
- Modify: `D:\myCodePycharm\Cool-Chic\coolchic\component\core\synthesis.py:18-75`
- Create: `D:\myCodePycharm\Cool-Chic\test\test_qat_synthesis.py`

**Interfaces:**
- Produces: `fixed_step_fake_quant(x: Tensor, q_step: float) -> Tensor`.
- Produces: `SynthesisConv2d.enable_fixed_qat(weight_q_step: float, bias_q_step: float) -> None`.
- Produces: `SynthesisConv2d.disable_fixed_qat() -> None`.
- Produces: `SynthesisConv2d.fixed_qat_enabled: bool` read-only property.

- [ ] **Step 1: Write failing forward-value and gradient tests**

```python
import unittest

import torch

from coolchic.component.core.synthesis import fixed_step_fake_quant


class TestFixedStepFakeQuant(unittest.TestCase):
    def test_forward_matches_coolchic_hard_quantization(self):
        x = torch.tensor([-0.31, -0.25, -0.01, 0.0, 0.24, 0.26, 0.76])
        actual = fixed_step_fake_quant(x, 0.25)
        expected = torch.round(x / 0.25) * 0.25
        self.assertTrue(torch.equal(actual, expected))

    def test_backward_is_straight_through(self):
        x = torch.tensor([-0.31, 0.24, 0.76], requires_grad=True)
        fixed_step_fake_quant(x, 0.25).sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.ones_like(x)))

    def test_non_positive_step_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            fixed_step_fake_quant(torch.ones(1), 0.0)
```

- [ ] **Step 2: Run the focused test and confirm it fails for the missing API**

Run from `D:\myCodePycharm\Cool-Chic`:

```powershell
python -m unittest test.test_qat_synthesis -v
```

Expected: import failure for `fixed_step_fake_quant`.

- [ ] **Step 3: Implement the pure STE helper**

Add beside `SynthesisConv2d` in `synthesis.py`:

```python
def fixed_step_fake_quant(x: Tensor, q_step: float) -> Tensor:
    q_step = float(q_step)
    if not math.isfinite(q_step) or q_step <= 0:
        raise ValueError(f"q_step must be finite and positive. Found {q_step}")
    q = x.new_tensor(q_step)
    hard = torch.round(x / q) * q
    return x + (hard - x).detach()
```

- [ ] **Step 4: Write failing synthesis opt-in tests**

```python
import torch.nn.functional as F

from coolchic.component.core.synthesis import SynthesisConv2d


class TestSynthesisConv2dQAT(unittest.TestCase):
    def test_layer_uses_separate_weight_and_bias_steps(self):
        layer = SynthesisConv2d(1, 1, 1)
        with torch.no_grad():
            layer.weight.fill_(0.26)
            layer.bias.fill_(0.13)
        x = torch.ones(1, 1, 1, 1)
        original_weight = layer.weight.detach().clone()
        original_bias = layer.bias.detach().clone()

        layer.enable_fixed_qat(weight_q_step=0.25, bias_q_step=0.1)
        actual = layer(x)
        expected = F.conv2d(x, torch.tensor([[[[0.25]]]]), torch.tensor([0.1]))

        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(layer.weight, original_weight))
        self.assertTrue(torch.equal(layer.bias, original_bias))
        layer.disable_fixed_qat()
        self.assertFalse(layer.fixed_qat_enabled)
```

- [ ] **Step 5: Add disabled-by-default per-layer state and use local fake tensors in `forward()`**

Use private Python attributes so QAT state does not enter `state_dict()`:

```python
self._fixed_qat_weight_step: Optional[float] = None
self._fixed_qat_bias_step: Optional[float] = None

@property
def fixed_qat_enabled(self) -> bool:
    return self._fixed_qat_weight_step is not None

def enable_fixed_qat(self, weight_q_step: float, bias_q_step: float) -> None:
    weight_q_step = float(weight_q_step)
    bias_q_step = float(bias_q_step)

def disable_fixed_qat(self) -> None:
    self._fixed_qat_weight_step = None
    self._fixed_qat_bias_step = None
```

In `forward()`, preserve padding/residual behavior and only replace the tensors passed to `F.conv2d`:

```python
weight = self.weight
bias = self.bias
if self.fixed_qat_enabled:
    weight = fixed_step_fake_quant(weight, self._fixed_qat_weight_step)
    bias = fixed_step_fake_quant(bias, self._fixed_qat_bias_step)
y = F.conv2d(padded_x, weight, bias, groups=self.groups)
```

- [ ] **Step 6: Run the unit tests**

```powershell
python -m unittest test.test_qat_synthesis -v
```

Expected: all fake-quant forward, gradient, validation, and state-preservation tests pass.

- [ ] **Step 7: Commit only Task 1 files**

```powershell
git add coolchic/component/core/synthesis.py test/test_qat_synthesis.py
git diff --cached --check
git commit -m "feat: add fixed-step synthesis fake quantization"
```

### Task 2: QAT Configuration and Parameter-State Isolation

**Files:**
- Create: `D:\myCodePycharm\Cool-Chic\coolchic\nnquant\qat.py`
- Create: `D:\myCodePycharm\Cool-Chic\test\test_qat.py`

**Interfaces:**
- Consumes: `SynthesisConv2d.enable_fixed_qat()` and `disable_fixed_qat()` from Task 1.
- Produces: `QATConfig.from_env(env: Mapping[str, str] = os.environ) -> QATConfig`.
- Produces private helpers `_target_named_parameters`, `_restore_float_masters`, `_snapshot_target_parameters`, `_restore_target_parameters`, `_set_fake_quant`, `_hard_quantize_targets`, `_freeze_for_qat`, and `_restore_requires_grad`.

- [ ] **Step 1: Write failing configuration tests**

```python
class TestQATConfig(unittest.TestCase):
    def test_defaults_are_approved_values(self):
        cfg = QATConfig.from_env({})
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.iterations, 1000)
        self.assertEqual(cfg.learning_rate, 2e-5)
        self.assertEqual(cfg.patience, 300)
        self.assertEqual(cfg.validation_frequency, 50)
        self.assertEqual(cfg.byte_frequency, 100)
        self.assertEqual(cfg.minimum_iterations, 200)

    def test_disabled_qat_ignores_stale_invalid_tuning_values(self):
        cfg = QATConfig.from_env({'COMMA_QAT': '0', 'COMMA_QAT_ITERS': 'bad'})
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.iterations, 1000)

    def test_invalid_enabled_configuration_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "COMMA_QAT_ITERS"):
            QATConfig.from_env({"COMMA_QAT": "1", "COMMA_QAT_ITERS": "0"})
```

- [ ] **Step 2: Run the test and confirm the module is missing**

```powershell
python -m unittest test.test_qat.TestQATConfig -v
```

Expected: import failure for `coolchic.nnquant.qat`.

- [ ] **Step 3: Add immutable config parsing with exact environment names**

```python
@dataclass(frozen=True)
class QATConfig:
    enabled: bool = False
    iterations: int = 1000
    learning_rate: float = 2e-5
    patience: int = 300
    validation_frequency: int = 50
    byte_frequency: int = 100
    minimum_iterations: int = 200

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> "QATConfig":
        if env.get('COMMA_QAT', '0') != '1':
            return cls(enabled=False)
        cfg = cls(
            enabled=env.get("COMMA_QAT", "0") == "1",
            iterations=int(env.get("COMMA_QAT_ITERS", "1000")),
            learning_rate=float(env.get("COMMA_QAT_LR", "2e-5")),
            patience=int(env.get("COMMA_QAT_PATIENCE", "300")),
            validation_frequency=int(env.get("COMMA_QAT_FREQ_VALID", "50")),
            byte_frequency=int(env.get("COMMA_QAT_FREQ_BYTES", "100")),
        )
        cfg.validate()
        return cfg
```

Validation must name the offending environment variable and require finite positive LR and positive integer frequencies/iterations. Permit smoke tests with fewer than 200 total iterations; `minimum_iterations` is then effectively `min(200, iterations)`.

- [ ] **Step 4: Build a small fake FrameEncoder fixture and write state-isolation tests**

The fixture is an `nn.Module` containing `coolchic_enc["residue"].synthesis`, one non-target parameter, `full_precision_param` keys prefixed with `synthesis.`, and `nn_q_step.synthesis = DescriptorNN(weight=0.25, bias=0.1)`.

```python
def test_restore_float_masters_excludes_output_transform(self):
    encoder = make_fake_frame_encoder()
    output_before = encoder.coolchic_enc["residue"].synthesis.output_transform.weight.clone()
    expected_master = encoder.coolchic_enc["residue"].full_precision_param[
        "synthesis.main_branch.0.weight"
    ].clone()

    _restore_float_masters(encoder)

    self.assertTrue(torch.equal(
        encoder.coolchic_enc["residue"].synthesis.main_branch[0].weight,
        expected_master,
    ))
    self.assertTrue(torch.equal(
        encoder.coolchic_enc["residue"].synthesis.output_transform.weight,
        output_before,
    ))

def test_freeze_allowlist_and_restore_original_flags(self):
    encoder = make_fake_frame_encoder()
    original = _freeze_for_qat(encoder)
    trainable = [name for name, p in encoder.named_parameters() if p.requires_grad]
    self.assertTrue(trainable)
    self.assertTrue(all("residue.synthesis" in name for name in trainable))
    self.assertTrue(all("output_transform" not in name for name in trainable))
    _restore_requires_grad(encoder, original)
    self.assertEqual(
        {name: p.requires_grad for name, p in encoder.named_parameters()}, original
    )
```

- [ ] **Step 5: Implement target discovery using existing parameter names**

Target `synthesis.named_parameters()` entries except names beginning with `output_transform.`. Load masters from the already-existing `residue.full_precision_param` entries beginning with `synthesis.`, strip that prefix, filter the same output-transform names, and call `synthesis.set_param(filtered, strict=False)`.

Do not call `_load_full_precision_param()`: it resets all selected q-step and Exp-Golomb descriptors, which would violate fixed-step QAT.

- [ ] **Step 6: Implement fake enable/disable, snapshots, hard quantization, and trainability restoration**

Use `SynthesisConv2d` instances in `main_branch` plus `stabiliser_branch`; never include `output_transform`. Hard quantization is in-place under `torch.no_grad()` and uses each parameter name to choose `q_step.weight` or `q_step.bias`:

```python
param.copy_(torch.round(param / float(q_step)) * float(q_step))
```

Assert positive q-steps, at least one weight and bias target, no non-target trainable parameter, and unchanged q-step values.

- [ ] **Step 7: Run configuration and state tests**

```powershell
python -m unittest test.test_qat.TestQATConfig test.test_qat.TestQATState -v
```

Expected: all tests pass and `output_transform` remains untouched.

- [ ] **Step 8: Commit Task 2**

```powershell
git add coolchic/nnquant/qat.py test/test_qat.py
git diff --cached --check
git commit -m "feat: add fixed-step QAT state management"
```

### Task 3: Exact Candidate Evaluation and Byte Gate

**Files:**
- Modify: `D:\myCodePycharm\Cool-Chic\coolchic\nnquant\qat.py`
- Modify: `D:\myCodePycharm\Cool-Chic\test\test_qat.py`

**Interfaces:**
- Produces: `QATCandidate(step: int, master_params: OrderedDict[str, Tensor], rd_loss: float, payload_bytes: Optional[int], detailed_dist: Dict[str, float])`.
- Produces: `_evaluate_candidate(frame_encoder: FrameEncoder, frame: Frame, dist_weight: Dict[DISTORTION_METRIC, float], lmbda: float, reference_frame_encoder: Optional[FrameEncoder], step: int, measure_bytes: bool) -> QATCandidate`.
- Consumes the existing production `encode_network` API without changing its parameters or return values.

- [ ] **Step 1: Write a failing exact-reference and non-mutation test**

Patch `coolchic.nnquant.qat.encode_network` and make it assert object identity:

```python
def fake_encode_network(cc_enc, ref_cc_enc=None):
    self.assertIs(ref_cc_enc, expected_reference.coolchic_enc["residue"])
    return b"1234567", {"nn_n_bytes": 7}

master_before = _snapshot_target_parameters(encoder)
candidate = _evaluate_candidate(
    frame_encoder=encoder,
    frame=frame,
    dist_weight={"mse": 1.0},
    lmbda=0.01,
    reference_frame_encoder=expected_reference,
    step=0,
    measure_bytes=True,
)
self.assertEqual(candidate.payload_bytes, 7)
assert_parameter_snapshots_equal(master_before, _snapshot_target_parameters(encoder))
```

Also add a test where `encode_network` raises and verify master parameters and train/eval mode are restored in `finally`.

- [ ] **Step 2: Run the focused tests and confirm `_evaluate_candidate` is missing**

```powershell
python -m unittest test.test_qat.TestCandidateEvaluation -v
```

Expected: failure for the missing candidate API.

- [ ] **Step 3: Implement non-destructive hard candidate evaluation**

The evaluator must:

1. Snapshot current float target masters and current model mode.
2. Disable synthesis fake quantization temporarily.
3. Hard-quantize only target parameters with the fixed synthesis q-steps.
4. Set the frame encoder to eval and forward with `quantizer_type="hardround"`, `quantizer_noise_type="none"`, `AC_MAX_VAL=-1`.
5. If `measure_bytes`, call `encode_network` on the residue encoder and pass `reference_frame_encoder.coolchic_enc["residue"]` when present.
6. Add the constant estimated motion-network bits, if a motion encoder exists, to the exact residue payload bits for the RD calculation.
7. Call the existing `loss_function` with the candidate decoded image, candidate latent-rate dictionary, target image, distortion weights, lambda, exact candidate network bits, and `compute_logs=True`.
8. In `finally`, restore float target masters, restore fake quantization, and restore the prior model mode.

Use `len(payload)` as the exact production payload size; do not substitute `get_network_rate()` for the residue byte gate.

- [ ] **Step 4: Write and pass byte-valid ranking tests**

```python
baseline = QATCandidate(
    step=0, master_params=OrderedDict(), rd_loss=1.0,
    payload_bytes=100, detailed_dist={mse: 0.1},
)
smaller_better = QATCandidate(
    step=100, master_params=OrderedDict(), rd_loss=0.9,
    payload_bytes=99, detailed_dist={mse: 0.09},
)
larger_better = QATCandidate(
    step=200, master_params=OrderedDict(), rd_loss=0.8,
    payload_bytes=101, detailed_dist={mse: 0.08},
)

self.assertTrue(_is_better_candidate(smaller_better, baseline, baseline_bytes=100))
self.assertFalse(_is_better_candidate(larger_better, smaller_better, baseline_bytes=100))
```

An unmeasured candidate (`payload_bytes is None`) is loggable but never eligible for the saved best checkpoint.

- [ ] **Step 5: Run the full QAT unit module**

```powershell
python -m unittest test.test_qat -v
```

Expected: config, state, exact-reference, non-mutation, failure-cleanup, and byte-gate tests pass.

- [ ] **Step 6: Commit Task 3**

```powershell
git add coolchic/nnquant/qat.py test/test_qat.py
git diff --cached --check
git commit -m "feat: evaluate QAT checkpoints with exact network bytes"
```

### Task 4: SOAP QAT Loop and Quantization-Pipeline Hook

**Files:**
- Modify: `D:\myCodePycharm\Cool-Chic\coolchic\nnquant\qat.py`
- Modify: `D:\myCodePycharm\Cool-Chic\coolchic\component\video.py:33-43,342-371`
- Modify: `D:\myCodePycharm\Cool-Chic\test\test_qat.py`

**Interfaces:**
- Produces: `qat_model(frame_encoder: FrameEncoder, frame: Frame, dist_weight: Dict[DISTORTION_METRIC, float], lmbda: float, config: QATConfig, reference_frame_encoder: Optional[FrameEncoder] = None) -> FrameEncoder`.
- Changes: add final keyword parameter `network_rate_reference: Optional[FrameEncoder] = None` to the existing `encode_one_frame` signature, preserving every existing parameter and call.
- Consumes: Task 3 candidate evaluation and the existing `SOAP`, `loss_function`, and `rdoq_model` APIs.

- [ ] **Step 1: Write failing disabled-path and fallback tests**

```python
def test_disabled_qat_returns_same_model_without_mutation(self):
    encoder = make_fake_frame_encoder()
    before = clone_full_state(encoder)
    result = qat_model(encoder, frame, {"mse": 1.0}, 0.01, QATConfig(enabled=False))
    self.assertIs(result, encoder)
    assert_full_state_equal(before, clone_full_state(encoder))

def test_nonfinite_gradient_restores_step_zero_and_qsteps(self):
    encoder = make_fake_frame_encoder()
    master = clone_float_target_snapshot(encoder)
    qsteps_before = copy.deepcopy(
        encoder.coolchic_enc['residue'].nn_q_step.synthesis
    )
    flags_before = {
        name: parameter.requires_grad for name, parameter in encoder.named_parameters()
    }
    baseline = QATCandidate(
        step=0,
        master_params=master,
        rd_loss=1.0,
        payload_bytes=100,
        detailed_dist={'mse': 0.1},
    )
    nan_loss = SimpleNamespace(loss=torch.tensor(float('nan')))
    config = QATConfig(
        enabled=True, iterations=1, validation_frequency=1, byte_frequency=1
    )

    with patch('coolchic.nnquant.qat._evaluate_candidate', return_value=baseline), patch(
        'coolchic.nnquant.qat.loss_function', return_value=nan_loss
    ):
        qat_model(encoder, frame, {'mse': 1.0}, 0.01, config)

    assert_targets_equal_hard_quantized_master(encoder, master, qsteps_before)
    self.assertEqual(
        encoder.coolchic_enc['residue'].nn_q_step.synthesis, qsteps_before
    )
    self.assertFalse(any_qat_layer_enabled(encoder))
    self.assertEqual(
        {name: parameter.requires_grad for name, parameter in encoder.named_parameters()},
        flags_before,
    )
```

- [ ] **Step 2: Run the controller tests and confirm `qat_model` is missing**

```powershell
python -m unittest test.test_qat.TestQATController -v
```

- [ ] **Step 3: Implement SOAP setup aligned with `training/train.py`**

Use one target parameter group:

```python
optimizer = SOAP([{
    "params": target_parameters,
    "lr": config.learning_rate,
    "betas": (0.95, 0.95),
    "precondition_frequency": 10,
    "max_precond_dim": 256,
    "merge_dims": False,
    "precondition_1d": False,
    "weight_decay": 0.01,
}])
```

Keep the LR constant for v1. Clip target gradients exactly as current network training does:

```python
clip_grad_norm_(target_parameters, 0.1, norm_type=2.0, error_if_nonfinite=False)
```

- [ ] **Step 4: Implement the QAT loop with hard latents and fake-quantized synthesis**

The gradient forward is:

```python
frame_encoder.set_to_train()
out = frame_encoder.forward(
    reference_frames=[ref.data for ref in frame.refs_data],
    quantizer_noise_type="none",
    quantizer_type="hardround",
    AC_MAX_VAL=-1,
    flag_additional_outputs=False,
)
loss_out = loss_function(
    decoded_image=out.decoded_image,
    rate_latent_bit=out.rate,
    target_image=frame.data.data,
    dist_weight=dist_weight,
    lmbda=lmbda,
    total_rate_nn_bit=0.0,
    compute_logs=False,
)
```

Check the loss and every existing target gradient with `torch.isfinite` before `optimizer.step()`. Validate on the union of validation and byte intervals; call exact byte measurement only at step 0, `step % byte_frequency == 0`, and the final step. Only byte-measured candidates can replace the best.

Use `effective_minimum = min(config.minimum_iterations, config.iterations)` and stop when `step >= effective_minimum` and `step - best.step >= config.patience`.

- [ ] **Step 5: Implement safe selection, final hard quantization, and cleanup**

At entry, clone the two fixed q-step scalars. Restore float masters from `full_precision_param`, record original trainability flags, enable fake quantization, and evaluate step 0 with exact bytes.

On normal completion, restore the best floating-point master snapshot, clear fake quantization, and hard-quantize targets once with the saved fixed q-steps. On any caught runtime failure, print a traceback, restore step-0 masters, clear fake quantization, hard-quantize step 0, and return the model ready for RDOQ. A `finally` block restores original `requires_grad` flags and model mode. Assert q-steps are unchanged before return.

- [ ] **Step 6: Add logs and test the 1,000/300/50/100 schedule with small overrides**

Tests use `iterations=4`, `validation_frequency=1`, `byte_frequency=2`, `patience=3`, with mocked candidate payloads. Assert:

- step 0 always exists,
- the larger-byte candidate is rejected,
- the best eligible master snapshot is selected,
- final targets are hard-quantized,
- RDOQ is not called inside `qat_model`,
- stopping reason and fallback state are printed.

- [ ] **Step 7: Insert the QAT hook in `video.py`**

Keep the current order and quant-tax probe:

```python
frame_encoder = quantize_model(
    frame_encoder=frame_encoder,
    frame=frame,
    dist_weight=training_phase.dist_weight,
    lmbda=training_phase.lmbda,
)

try:
    qat_config = QATConfig.from_env()
except (TypeError, ValueError) as error:
    print(f'[qat] invalid configuration; keeping step 0: {error}', flush=True)
    qat_config = QATConfig(enabled=False)
if qat_config.enabled:
    frame_encoder = qat_model(
        frame_encoder=frame_encoder,
        frame=frame,
        dist_weight=training_phase.dist_weight,
        lmbda=training_phase.lmbda,
        config=qat_config,
        reference_frame_encoder=network_rate_reference,
    )

frame_encoder = rdoq_model(
    frame_encoder=frame_encoder,
    frame=frame,
    dist_weight=training_phase.dist_weight,
    lmbda=training_phase.lmbda,
)
```

Do not move the full-precision store, quant-tax snapshot, post-RDOQ tax measurement, final test, save, or decoded-frame write.

- [ ] **Step 8: Run QAT and compatibility unit tests**

```powershell
python -m unittest \
  test.test_qat_synthesis \
  test.test_qat \
  test.test_bitstream_version \
  test.test_coolchic_header_version \
  test.test_frame_reconstruction
```

Expected: all pass; the three existing compatibility modules still report 10 tests total.

- [ ] **Step 9: Commit Task 4 without staging unrelated work**

```powershell
git add coolchic/nnquant/qat.py coolchic/component/video.py test/test_qat.py
git diff --cached --check
git diff --cached --name-only
git commit -m "feat: run fixed-step QAT before RDOQ"
```

### Task 5: Reuse the Exact Delta Reference for QAT and Encoding

**Files:**
- Modify: `D:\myCodePycharm\Cool-Chic\cc_encode.py:10-25,450-535`
- Create: `D:\myCodePycharm\Cool-Chic\test\test_qat_reference.py`

**Interfaces:**
- Produces: `_load_delta_reference(coding_structure: CodingStructure, frame: Frame) -> Optional[FrameEncoder]` in `cc_encode.py`.
- Consumes: `find_same_role_reference`, `_get_frame_path_prefix`, and `load_frame_encoder` exactly as the current final-bitstream block does.
- Supplies the same object through `encode_one_frame(network_rate_reference=network_rate_reference)` and `encode_frame_with_reconstruction(ref_frame_encoder=network_rate_reference)`.

- [ ] **Step 1: Write failing reference-loader tests**

```python
class TestQATReferenceLoader(unittest.TestCase):
    @patch.dict(os.environ, {"COMMA_DELTA": "0"}, clear=False)
    def test_delta_off_does_not_load_reference(self):
        self.assertIsNone(_load_delta_reference(coding_structure, frame))

    @patch.dict(os.environ, {"COMMA_DELTA": "1"}, clear=False)
    @patch("cc_encode.load_frame_encoder")
    @patch("cc_encode.os.path.isfile", return_value=True)
    @patch("cc_encode.find_same_role_reference")
    def test_loads_exact_trained_same_role_reference(self, find_ref, _, load):
        find_ref.return_value.display_order = 3
        marker = object()
        load.return_value = marker
        self.assertIs(_load_delta_reference(coding_structure, frame), marker)
        load.assert_called_once_with("0003-frame_encoder.pt")
```

Add the no-same-role and missing-file cases; both must return `None`, matching the production full-network fallback.

- [ ] **Step 2: Run the test and confirm the helper is missing**

```powershell
python -m unittest test.test_qat_reference -v
```

- [ ] **Step 3: Extract the current reference logic into one module-level helper**

Move, rather than duplicate, the existing `find_same_role_reference`/file/load behavior. Print `[delta] reference frame = {ref_frame.frame_type}{ref_frame.display_order}` only when a reference is loaded.

- [ ] **Step 4: Load once and pass the identical object down both paths**

After `frame` and workdir are established, compute:

```python
network_rate_reference = _load_delta_reference(coding_structure, frame)
```

Pass it into `encode_one_frame` for QAT. Later, remove the duplicate reference-loading block and pass the same `network_rate_reference` to `encode_frame_with_reconstruction`. `COMMA_DELTA=0` continues to pass `None` everywhere.

- [ ] **Step 5: Run reference and delta unit tests**

```powershell
python -m unittest test.test_qat_reference coolchic.bitstream.neuralnet.test_delta -v
```

Expected: exact-reference helper cases and existing delta-selection tests pass.

- [ ] **Step 6: Stage only QAT/reference hunks from dirty `cc_encode.py`**

```powershell
git add test/test_qat_reference.py
git add -p cc_encode.py
git diff --cached --check
git diff --cached -- cc_encode.py
git commit -m "refactor: share exact network coding reference"
```

Do not stage existing unrelated bitstream-version or reconstruction changes.

### Task 6: Experiment Configuration, Documentation, and Banner Tests

**Files:**
- Modify: `experiments/coolchic_baseline/encode_video.py:24-130`
- Modify: `experiments/coolchic_baseline/run_one.sh:110-130`
- Modify: `experiments/coolchic_baseline/README.md`
- Create: `experiments/coolchic_baseline/test_qat_config.py`

**Interfaces:**
- Consumes only environment variables; subprocesses already inherit `os.environ`.
- Produces no new CLI arguments and does not alter run-folder naming automatically.

- [ ] **Step 1: Write a failing banner-default test**

Use `redirect_stdout`, `patch.dict`, and a minimal `argparse.Namespace` accepted by `_print_lever_banner`:

```python
def render_banner():
    args = argparse.Namespace(
        lmbda=80.0, n_frames=1, intra_pos='0', p_pos='', warmstart=False
    )
    stream = io.StringIO()
    with redirect_stdout(stream):
        _print_lever_banner(args, Path('.'), comma_mode=False)
    return stream.getvalue()


def test_banner_prints_approved_qat_defaults(self):
    with patch.dict(os.environ, {"COMMA_QAT": "1"}, clear=True):
        output = render_banner()
    self.assertIn("Fixed-q-step residue QAT", output)
    self.assertIn("iters=1000", output)
    self.assertIn("lr=2e-5", output)
    self.assertIn("patience=300", output)
    self.assertIn("valid=50", output)
    self.assertIn("bytes=100", output)

def test_banner_reports_qat_off_by_default(self):
    with patch.dict(os.environ, {}, clear=True):
        self.assertIn('off', render_banner())
```

- [ ] **Step 2: Run and confirm the banner lacks QAT**

```powershell
python -m unittest experiments.coolchic_baseline.test_qat_config -v
```

- [ ] **Step 3: Add the QAT line to `_print_lever_banner`**

Read the exact same six environment names/default strings as `QATConfig`. Print one consolidated line containing enabled state and all defaults. Treat only `COMMA_QAT=1` as enabled.

- [ ] **Step 4: Document environment forwarding and fresh-run behavior**

Update `run_one.sh` comments and concise config output. Do not add separate exports: caller-provided `COMMA_QAT*` values already flow through `subprocess.run`.

In the README document:

```bash
# matched control
COMMA_QAT=0 bash experiments/coolchic_baseline/run_one.sh ~/Cool-Chic 80 32 auto ippp eval comma qat_off

# 500-step pilot
COMMA_QAT=1 COMMA_QAT_ITERS=500 bash experiments/coolchic_baseline/run_one.sh ~/Cool-Chic 80 32 auto ippp eval comma qat_500

# approved default
COMMA_QAT=1 bash experiments/coolchic_baseline/run_one.sh ~/Cool-Chic 80 32 auto ippp eval comma qat_1000
```

State explicitly that QAT is applied only while training a fresh frame. A completed/resumed frame is skipped by existing behavior, so QAT-on experiments need a fresh run tag.

- [ ] **Step 5: Run experiment-side unit tests**

```powershell
python -m unittest \
  experiments.coolchic_baseline.test_qat_config \
  experiments.coolchic_baseline.test_archive_rate
```

Expected: banner tests and the three archive-rate tests pass.

- [ ] **Step 6: Commit only Task 6 changes in the challenge repository**

Because `README.md` already has uncommitted edits, stage it interactively:

```powershell
git add experiments/coolchic_baseline/encode_video.py \
        experiments/coolchic_baseline/run_one.sh \
        experiments/coolchic_baseline/test_qat_config.py
git add -p experiments/coolchic_baseline/README.md
git diff --cached --check
git commit -m "docs: expose Cool-Chic QAT experiment controls"
```

### Task 7: Regression Gates and Cloud Pilot Handoff

**Files:**
- Verify only; modify the owning task's files if a regression is found.

**Interfaces:**
- Consumes the completed QAT implementation, current bitstream versioning, exact-reference delta coding, frame reconstruction, and temporary-ZIP scorer.
- Produces reproducible commands and measured QAT-off/QAT-on artifacts; it does not SCP automatically.

- [ ] **Step 1: Run all focused local Cool-Chic tests**

From `D:\myCodePycharm\Cool-Chic`:

```powershell
python -m unittest \
  test.test_qat_synthesis \
  test.test_qat \
  test.test_qat_reference \
  test.test_bitstream_version \
  test.test_coolchic_header_version \
  test.test_frame_reconstruction \
  coolchic.bitstream.neuralnet.test_delta
```

Expected: all tests pass with no warnings about changed q-steps or lingering fake quantization.

- [ ] **Step 2: Run challenge-side regression tests**

From `D:\myCodePycharm\comma_video_compression_challenge`:

```powershell
python -m unittest \
  experiments.coolchic_baseline.test_qat_config \
  experiments.coolchic_baseline.test_archive_rate
```

- [ ] **Step 3: Review both diffs for scope and compatibility**

```powershell
git -C D:\myCodePycharm\Cool-Chic diff --check
git -C D:\myCodePycharm\Cool-Chic status --short
git diff --check
git status --short
```

Confirm there is no decoder or bitstream-format change and no Muon code in this feature.

- [ ] **Step 4: Provide, but do not run, SCP commands**

Provide commands that copy only the changed Cool-Chic source/tests and challenge experiment files to their matching cloud paths. Do not create or use a password-bearing command; let the user's SSH client prompt or use their configured key.

- [ ] **Step 5: Run a two-iteration cloud smoke test in a fresh folder**

Use `COMMA_QAT=1 COMMA_QAT_ITERS=2 COMMA_QAT_FREQ_VALID=1 COMMA_QAT_FREQ_BYTES=1` and a one-frame/debug configuration. Required log sequence:

```text
quantize_model q-step search
[qat] step 0
[qat] step 1
[qat] step 2
[qat] selected step=N stop_reason=completed
Start discrete RDOQ
```

Then decode the resulting bitstream and require success.

- [ ] **Step 6: Run the matched 500/1,000-step pilot**

Use fresh run tags and identical seed/lambda/data for `QAT_OFF`, `QAT_500`, and `QAT_1000`. Score each completed run with:

```bash
python experiments/coolchic_baseline/score_coolchic.py --run <RUN_DIR>
```

Record raw bitstream bytes, temporary-ZIP bytes, segmentation, pose, total score, selected QAT step per frame, and fallback count.

- [ ] **Step 7: Apply the approved acceptance rule**

Retain QAT only if it recovers at least 30% of the measured residue-synthesis quantization tax on probe frames, does not increase the matched final ZIP size, preserves pose stability, and decodes byte-exactly. Prefer 500 steps if its result is equivalent to 1,000; test 2,000 only if 1,000 is still improving without rate growth.
