# coolchic_base — task-aware Cool-Chic (overfitted learned codec)

A learned video codec for the comma challenge built on **Cool-Chic 5.0.1**
(Orange's overfitted image/video codec), with the distortion swapped from
MSE to the challenge's own **SegNet + PoseNet** losses. To our knowledge this
is the first Cool-Chic–based entry for this challenge; every leaderboard neural
submission so far is HNeRV-lineage.

## Result

| metric | value | score term |
|---|---|---|
| SegNet distortion | 0.00514603 | 100·seg = 0.5146 |
| PoseNet distortion | 0.00077760 | √(10·pose) = 0.0882 |
| rate (extrapolated) | 0.02904377 | 25·rate = 0.7261 |
| **Final score** | | **1.3289** |

References: ffmpeg baseline 4.39 · hnerv_muon ~0.20 · SOTA ~0.17–0.19.
This **beats the entire published ffmpeg grid-search frontier** (best ≈ 3.0)
while not reaching the INR frontier (see *Positioning*).

> **STATUS — proof of concept.** The shipped `archive.zip` is a **32-frame**
> bitstream (29,079 B) coded from the first 32 frames of `videos/0.mkv`. The
> score above uses the challenge formula with the rate **extrapolated to the
> full 1200-frame video** (29,079 → 1,090,462 B). This is **not yet a full
> valid submission** — a full run means re-encoding all 1200 frames (see
> *Reproduce → full*). It is packaged here to document the method and the
> measured operating point.

## Method

- **Code at the judges' resolution (512×384), not camera resolution.** Both
  SegNet and PoseNet downscale their input to 512×384, so detail above that is
  invisible to the metric. The decoder codes at 512×384; `inflate.py`
  bicubic-upsamples to 874×1164 only to satisfy the output format.
- **IPPP low-delay GOP.** Frame 0 intra, every later frame a P-frame
  referencing its predecessor — so the even (PoseNet-only) frame of each pair
  is decoded before its odd partner, making the PoseNet pair-loss well-defined.
- **`--tune=comma` distortion** (`coolchic_mods/comma.py`): the decoded frames
  are rendered through the exact evaluation chain (differentiable YUV→RGB,
  bicubic upsample, straight-through rounding) into the frozen judges. Odd
  frames: `100·seg_margin_surrogate + √(10·pose_MSE)`; even frames: an MSE
  anchor. Targets are the SegNet argmax labels + PoseNet vectors of the
  originals (`encode/prep_targets.py`).
- **Native rate–distortion training.** Cool-Chic's learned autoregressive
  entropy model + `D + λ·R` objective; here `λ = 40`.

The decode path is **stock Cool-Chic** — the comma modifications are
encode-side (training) only; the decoder/entropy-coder are unchanged.

## Files

| path | role |
|---|---|
| `archive.zip` | charged payload — the Cool-Chic bitstream (`bitstream.cool`, 32-frame PoC) |
| `inflate.sh`, `inflate.py` | harness decode: bitstream → YUV 512×384 → RGB → bicubic 874×1164 → `<base>.raw` |
| `compress.sh` | reference reproduction steps (not run by the harness) |
| `encode/` | the 1.33-version encode kit (challenge commit `5879304`) |
| `coolchic_mods/comma.py` | the `--tune=comma` judge-loss metric (Cool-Chic commit `f59019e`) |
| `coolchic_mods/integration.patch` | the 3 integration edits into stock Cool-Chic (loss/parsecli/video) |

## Dependencies

Decode requires the **Cool-Chic 5.0.1** stack: `torch>=2.11`, `constriction`,
and the `coolchic` package. Point `COOLCHIC_REPO` at a checkout (default
`~/Cool-Chic`); check out commit `f59019e` to get the comma modifications
already applied, or apply `coolchic_mods/` to stock 5.0.1. The Cool-Chic
package is **not bundled** here (it is large and pip/GitHub-installable).

## Reproduce

Quick (this 32-frame PoC): see `compress.sh`.

Full submission (all 1200 frames):
```bash
# comma-compress env: prep the whole video + targets
bash  encode/prep_yuv.sh    1200 eval
python encode/prep_targets.py --n-frames 1200
# coolchic env: encode all frames (long; offline/untimed)
COMMA_CHALLENGE_ROOT=<challenge_root> \
  bash encode/run_one.sh $COOLCHIC_REPO 40 1200 <gpu> ippp eval comma
# package the resulting bitstream.cool as archive.zip
```

## Positioning (honest)

- **Beats classical.** At distortion 0.60 / rate 0.73 the operating point sits
  below-left of the *entire* ffmpeg grid search — lower distortion than every
  ffmpeg setting.
- **Does not reach the INR frontier (~0.19).** The gap is structural: Cool-Chic
  is a per-frame image codec, so it re-instantiates a full codec (latents +
  networks + container) for every frame. Extrapolated, that is ~1 MB for the
  video vs. hnerv's amortized ~173 KB — the rate term (0.73) dominates and has
  a per-frame floor that λ cannot cross. The distortion side is competitive;
  the rate side is limited by the lack of cross-frame amortization.
- **What it demonstrates:** a working task-metric-driven learned codec (the
  first Cool-Chic-based entry here), with the judge-loss port and the
  resolution/GOP exploits validated end to end.
