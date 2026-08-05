# Cool-Chic vanilla baseline on the challenge metric

Calibration experiment: can stock Cool-Chic 5.0 (MSE-trained, full-res) reach
a competitive contest score at the SOTA rate region (~176 KB / 60 s ≈ 0.0012 bpp)?

Two conda envs are involved:
- `coolchic` (torch>=2.11, Cool-Chic requirements.txt) — encode/decode
- `comma-compress` (repo environment.yml) — prep + scoring

## Workflow (server)

```bash
# 0) one-time: Cool-Chic env
git clone https://github.com/Orange-OpenSource/Cool-Chic.git ~/Cool-Chic
conda create -n coolchic python=3.12 -y
conda activate coolchic
pip install -r ~/Cool-Chic/requirements.txt
cd ~/Cool-Chic && python -m test.sanity_check

# 1) prep YUV (comma-compress env; ffmpeg + videos/0.mkv)
conda activate comma-compress
bash experiments/coolchic_baseline/prep_yuv.sh 200

# 2) pilot: timing + rate calibration (coolchic env, ~1-2 h)
conda activate coolchic
bash experiments/coolchic_baseline/run_one.sh ~/Cool-Chic 0.02 32 3

# 3) score the pilot (comma-compress env)
conda activate comma-compress
python experiments/coolchic_baseline/score_coolchic.py \
    --run experiments/coolchic_baseline/runs/lmbda_0.02_n32

# 4) sweep (one lambda per GPU, coolchic env)
bash experiments/coolchic_baseline/run_one.sh ~/Cool-Chic 0.005 200 1
bash experiments/coolchic_baseline/run_one.sh ~/Cool-Chic 0.02  200 2
bash experiments/coolchic_baseline/run_one.sh ~/Cool-Chic 0.08  200 3
# ... then score each run as in step 3.
```

## Interpretation

- `score_coolchic.py` puts the current `bitstream.cool` into a temporary
  Deflate-9 ZIP, extrapolates that archive size from N coded frames to the full
  1200, and prints `seg`, `pose`, `rate`, and the contest score. Raw and ZIP
  byte counts accumulate in `results_v2.csv`; the temporary ZIP is not kept.
- References: ffmpeg baseline 4.39, hnerv_muon ~0.20, SOTA 0.1885.
- The first 200 frames are an easier-than-average sample (single scene, no
  cut); treat the score as optimistic by some margin.
- Decision rule: score near ~0.3-0.5 at the SOTA rate -> proceed with the
  384x512 coding arm and the judge-loss port. Score >1 at all reachable
  rates -> the rate floor / MSE misallocation verdict, document and stop.

## Exact references and bitstream versions

- New Cool-Chic streams start with the COOLCHIC magic plus format version 2.
  Version 2 carries the temporal weight-delta flags in each Cool-Chic header.
- An unprefixed stream is decoded as stock legacy version 1, whose headers have
  no delta field. Pre-version experimental delta streams are intentionally not
  auto-detected; use the older checkout that created them.
- With SKIP_PERFRAME_DECODE=1 (the default), the encoder reuses the network and
  latent values already decoded from serialized bytes, composes the exact
  deployed frame once, and overwrites the per-frame decoded YUV. The next frame
  therefore trains against the decoder-exact reference in O(N) total.
- Set SKIP_PERFRAME_DECODE=0 only for a short validation run. It additionally
  decodes the complete prefix after every frame and fails on any difference,
  which is byte-exact but O(N^2).
- test_delta_e2e.py is the cloud gate for full-vs-delta losslessness and
  incremental-vs-one-shot reconstruction parity.
