<!-- SPDX-License-Identifier: MIT -->

# rhnerv_latent_polish

Exact-score-gated **latent polish** of the current #1 payload
([#112](https://github.com/commaai/comma_video_compression_challenge/pull/112)
`rhnerv_comma`). Decoder weight bytes are byte-identical to PR
[#101](https://github.com/commaai/comma_video_compression_challenge/pull/101)/[#95](https://github.com/commaai/comma_video_compression_challenge/pull/95),
the FEC6 selector to PR
[#110](https://github.com/commaai/comma_video_compression_challenge/pull/110),
and the container/entropy coder and decode chain to PR #112. Two changes,
both confined to the stored latent codes:

1. **Sidecar folded in**: PR #101's 607-byte latent-correction sidecar is
   absorbed into the stored 8-bit latent codes and dropped from the archive.
2. **~1,565 one-step latent code adjustments**, found by a discrete search
   in which every candidate ±1-step adjustment was scored with the official
   SegNet/PoseNet evaluators and the real re-encoded archive size, and kept
   only if the exact contest score improved. A single adjustment affects only
   its own frame pair, so candidates were evaluated exactly (600 per batched
   render) rather than estimated; accepted sets compose exactly across pairs.
   No gradient step was ever applied to stored values; no new training.

The search plateaued after 14 rounds (a final full sweep of all 33,600
possible one-step adjustments found zero improving moves).

## Archive identity

| Field | Value |
|---|---|
| Score (CPU, full precision) | `0.189227` = 100·seg + sqrt(10·pose) + 25·rate |
| seg / pose | `0.00054527` / `0.00002943` |
| rate | `0.00470179` (176,531 / 37,545,489) |
| Archive bytes | `176531` (#112: 177,136; −605 B, seg −2.7%, pose unchanged) |
| Archive SHA-256 | `ab73259395f9f87e0ca62623746095208bca7d33b272c6740336e69ca73fc01e` |
| Member SHA-256 | `5f0ade2878c10ab71c1fbcaa9c16755c00a882237df9374a8e325dac37e57e06` |
| ZIP members | 1 (`x`, `compression_type=0` ZIP_STORED, 176,431 bytes) |
| Member layout | ctx container (7-B header + decoder 161,104 + latents + selector); **no trailing sidecar** |
| Inflate runtime deps | `numpy`, `torch`, `constriction` (harness base env) |
| Inflate GPU required | no (device pinned to CPU) |

## Quick reproducibility check (CPU only)

```bash
mkdir -p /tmp/data /tmp/out
unzip -oq archive.zip -d /tmp/data
echo "0.mkv" > /tmp/list.txt
bash inflate.sh /tmp/data /tmp/out /tmp/list.txt
sha256sum /tmp/out/0.raw   # see expected_output.sha256 (machine-dependent LSBs, see #112 README)
```

Or the full harness:

```bash
bash evaluate.sh --submission-dir ./submissions/rhnerv_latent_polish --device cpu
```

`F.interpolate(mode='bicubic')` LSBs differ across CPU microarchitectures
(see #112's README); metrics reproduce machine-independently.

## Files

| Path | Role |
|---|---|
| `inflate.sh`, `inflate.py` | Contest-runtime decoder (#112 chain minus the sidecar branch). |
| `compress.sh`, `compress.py` | Deterministic encoder: re-runs the ctx coder on `encoder/` inputs to rebuild `archive.zip` byte-for-byte (asserts member + archive SHA-256). |
| `encoder/decoder_streams.bin` | Raw HNeRV decoder weight streams, verbatim #101/#95 (frozen). |
| `encoder/selector_payload.bin` | Raw FEC6 selector wire payload, verbatim #110 (frozen). |
| `encoder/polished_latent_raw.bin` | This submission's polished per-pair latent payload (sidecar folded in; ~1,565 verified ±1 code steps). |
| `codec_ctx.py` | #112's context-modeled range coder (verbatim). |
| `codec.py` | #101 tensor reconstruction (verbatim #112 copy). |
| `frame_selector.py` | #110 FEC6 selector module (verbatim). |
| `model.py` | PR #95 HNeRV decoder (verbatim). |
| `expected_output.sha256` | Canonical CPU decode SHA on the author's machine. |
| `THIRD_PARTY_NOTICES.md` | Upstream attribution (PR #95 / #98 / #101 / #110 / #112). |

## Method note

The starting payload sits at a sharp optimum: gradient fine-tuning (even
entropy-regularized QAT with straight-through rounding) and any zero-shot
re-quantization measurably worsen the exact score, and one-step *weight*
code moves are all individually harmful (verified exhaustively down to
single flips). The stored latents are the one section with residual slack,
and only an exact-acceptance search can harvest it safely: distortion
deltas are computed exactly per candidate (pair-local rendering), byte
deltas by re-encoding with the real coder at accept time, and every
accepted set is re-verified end to end before being kept.
