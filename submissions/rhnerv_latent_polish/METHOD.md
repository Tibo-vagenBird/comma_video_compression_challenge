<!-- SPDX-License-Identifier: MIT -->

# How the latent polish works, step by step

This document explains, from the bytes up, what this submission changes
relative to PR #112 and what every accepted "click" means, with a worked
dummy example.

## 1. What the archive stores

The 176,531-byte `archive.zip` holds one member, a ctx-coded container
(PR #112's format) with three sections:

| Section | Size (coded) | Content | Status here |
|---|---|---|---|
| decoder | 161,104 B | 229,014 quantized weight bytes of a 229K-param HNeRV decoder (28 tensors, each `uint8 codes × fp16 scale`) | **frozen** (byte-identical to #101/#95) |
| latents | ~15.1 KB | 600 pairs × 28 dims of `uint8` codes + a 112-byte header (28 fp16 mins + 28 fp16 scales) | **this submission's change** |
| selector | 248 B | per-pair choice among 16 decode-side frame-0 transforms (#110's FEC6) | frozen (byte-identical to #110) |

PR #112 additionally carried a 607-byte "sidecar" of sub-grid latent
corrections; this submission folds their effect into the latent codes and
drops the section.

## 2. What a latent code is, and what one "click" means

Each frame pair `p ∈ [0,600)` is rendered from 28 numbers
`z[p,0..27]`. Dimension `d` is stored as an 8-bit code on a per-dim grid:

```
value(p,d) = min[d] + code[p,d] * scale[d]        code ∈ {0,…,255}
```

`min[d]` and `scale[d]` are fp16 constants in the header — the grid is the
storage format, so the **smallest expressible change** to any latent is one
code step ("one click"), worth exactly `scale[d]` in latent units. (This is
also why ±0.5-step moves don't exist: they aren't representable without
re-adding a sidecar, whose byte cost exceeds its value — measured.)

A latent dimension is an abstract feature the decoder learned for that
2-frame chunk (there is no human label like "brightness"; it's whatever the
training run made it mean). A click nudges that feature by its minimum
increment, which slightly shifts the rendered pixels of **that pair only**.

### Dummy example (illustrative numbers)

Say dim 9 has `min = −1.20`, `scale = 0.02`, and pair 217 stores
`code[217,9] = 143`:

```
value = −1.20 + 143·0.02 = 1.66
+1 click → code 144 → value 1.68
```

Decoding pair 217 with 1.68 instead of 1.66 changes its two rendered
frames imperceptibly — but the metric is not human perception. It's two
frozen judges:

- **SegNet** labels every pixel of frame 1 into 5 classes; the seg
  distortion of the pair is the fraction of its 196,608 (384×512) label
  pixels that disagree with the labels of the *original* frame.
- **PoseNet** reads both frames and outputs 6 motion numbers; pose
  distortion is their squared error vs. the original pair's outputs.

A typical accepted click flips **~2–5 label pixels** of frame 217 back
into agreement (e.g. pair seg 0.00058 → 0.00056) and leaves pose
essentially unchanged. Because the final score averages over 600 pairs and
weights seg by 100:

```
Δscore ≈ 100 · (−0.00002)/600  ≈  −3·10⁻⁶ per click
```

Fifteen hundred such clicks ≈ −0.005 … in practice, with overlaps and byte
effects, the measured total was −0.0019 → −0.0026 on the official CPU axis.

The click also changes bytes slightly: latent codes are stored as
*temporal deltas* (`code[p,d] − code[p−1,d]`), entropy-coded. Moving
`code[217,9]` changes two deltas (at p=217 and p=218), each worth a few
bits under the coder's statistics — usually ±0…4 bits, i.e. ±0…0.5 bytes,
i.e. |Δscore| ≤ 3·10⁻⁷ (1 byte = 25/37,545,489 = 6.66·10⁻⁷ score). Byte
effects are therefore second-order but not ignored (step 5 below).

## 3. Why one click affects exactly one pair (the key structural fact)

The decode chain is: `z[p] → HNeRV decoder → 2 frames (384×512) → bicubic
upsample to 874×1164 → #98 channel biases → clamp/round → FEC6 transform →
judged`. Pair `p`'s frames depend only on row `z[p]` — no cross-frame
state. And the contest metric is a **mean over the 600 pairs** of per-pair
judgments. Consequences:

1. The exact effect of a candidate click on the total score can be
   computed by re-rendering **one pair** (plus its byte delta).
2. Effects of accepted clicks on **different pairs add up exactly** — no
   interaction terms, no surprises when a set is applied jointly.

## 4. The search, step by step

Let `S(payload)` be the exact contest score: re-encode the archive with
the real coder (bytes term) + run both judges on all 600 decoded pairs
(distortion terms). One full `S` evaluation ≈ seconds (GPU) / minutes (CPU).

Each **round**:

1. **Baseline**: decode the current payload, record per-pair seg/pose for
   all 600 pairs, and the exact coded size.
2. **Sweep (diagonal batching)**: for every dimension `d` (28) and step
   `δ ∈ {+1,−1,+2,−2}` (112 combinations): build a candidate latent table
   where **every pair simultaneously** has `code[p,d] += δ` (its own copy —
   pair-locality makes this legal), decode all 600 pairs once, and record
   per-pair seg/pose. One render therefore scores 600 independent
   candidates *exactly*; the full sweep scores all 600×28×4 = 67,200
   possible clicks.
3. **Byte estimate**: for each candidate, the entropy change of the two
   affected temporal deltas under the coder's empirical statistics.
4. **Candidate list**: every click whose exact distortion delta + byte
   estimate is a net improvement.
5. **Selection**: sort by predicted gain; take at most **one click per
   pair** per round (clicks on the *same* pair can interact through the
   decoder's nonlinearity; clicks on different pairs cannot).
6. **Exact verification**: apply the whole selected set, re-encode with
   the real coder, re-judge all 600 pairs, compute `S`. Accept only if `S`
   strictly improves; on failure, bisect the set (never happened for
   latent sets — step 3's estimate is the only approximate ingredient).
7. **Bank**: write the accepted member to disk; it is a valid, shippable
   archive at every point in time.

Rounds repeat until a full sweep yields zero accepted clicks (plateau).
Multi-click-per-pair configurations are reached *across* rounds: a pair
can receive one click per round on different (or the same) dims.

## 5. Why "exact-gated" is the whole point

Three cheaper alternatives were tried first and all failed, which is what
motivated this design:

- **Zero-shot re-quantization** of weight tensors (7-bit, deadzone):
  saves 1.5–4 KB but costs 16–25× more in seg distortion. The payload
  sits at a sharp optimum tuned by #101's quantization-aware training.
- **Gradient fine-tuning** (entropy-regularized QAT, straight-through
  rounding): the surrogate loss improved while the *true* seg metric
  doubled — per-pixel-margin surrogates optimize the wrong thing near
  this optimum, and the drift is systematic, not a step-size issue.
- **Weight-code clicks**: the same search as §4 applied to the 229K
  decoder weight codes, with gradient-ranked proposals, bisected down to
  *single* clicks — every one rejected. The decoder weights are at a
  strict discrete local optimum; the latents were the one section with
  residual slack.

The search accepts nothing on prediction: every kept change is verified
against the exact score. It is monotone by construction.

## 6. CPU axis and ±2 steps

The leaderboard evaluates on CPU. GPU and CPU renders differ in the least
significant bits of the bicubic upsample, which is enough to flip a few
borderline judge-pixels: clicks selected on the GPU axis lost ≈0.0009 of
their measured gain when re-scored on CPU. The final polish therefore
runs the sweep **on the CPU axis directly** (fp32, fixed batch layout),
which also automatically re-audits earlier GPU-selected clicks — undoing
any click is just a click in the opposite direction, and the sweep
proposes those too. ±2-step candidates were added for the same reason:
some pairs want a two-click move whose one-click midpoint is worse.

## 7. Accounting (official `evaluate.py`, CPU)

| | PR #112 | this submission |
|---|---|---|
| SegNet distortion | 0.00056032 | see `report_cpu.txt` |
| PoseNet distortion | 0.00002943 | see `report_cpu.txt` |
| archive bytes | 177,136 | 176,531 (−605: sidecar folded, +2 B latent entropy) |
| net changed latent codes | — | 1,802 of 16,800 (577 of 600 pairs touched) |
| decoder weight bytes | — | byte-identical to #101/#95 |
| selector | — | byte-identical to #110 |

No training, no new model, no change to any decode-side constant: the
entire submission is "which 8-bit codes to store", chosen one verified
click at a time.
