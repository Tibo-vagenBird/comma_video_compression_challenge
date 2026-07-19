# comma.ai Video Compression Challenge — Context Briefing

> Purpose: context handoff for Claude Code. I'm applying to the comma.ai internship via their
> lossy video compression challenge. My background is visual localization (ACE-style,
> per-scene overfit neural representations). This file summarizes the challenge, the SOTA
> submission I'm studying, and my planned attack. Local repo work happens from here.

## Key links

- Challenge repo: https://github.com/commaai/comma_video_compression_challenge
- SOTA submission I'm studying: https://github.com/a12dongithub/comma_video_compression_challenge/tree/submission/rhnerv_latent_polish/submissions/rhnerv_latent_polish
- Leaderboard: https://comma.ai/leaderboard
- Dataset (comma2k19 + compression_challenge test videos): https://huggingface.co/datasets/commaai/comma2k19
- HNeRV paper (CVPR 2023): https://arxiv.org/abs/2304.02633
- HiNeRV paper (NeurIPS 2023): https://arxiv.org/abs/2306.09818
- NOTE: do not confuse with the older *lossless* commaVQ token challenge
  (https://github.com/commaai/commavq) — different problem, ended 2024.

## Challenge definition

- Task: lossy compression of short raw dashcam video clips (comma2k19 driving footage).
- Scoring (lower is better):
  `Final score = 100 * segnet_dist + sqrt(10 * posenet_dist) + 25 * rate`
  - `segnet_dist` / `posenet_dist`: disagreement of comma's SegNet / PoseNet models
    between original and reconstructed frames — NOT PSNR, not human-perceptual quality.
  - `rate` = compressed size / original uncompressed size.
- Official ffmpeg baseline (`submissions/baseline_fast`): PoseNet dist 0.380,
  SegNet dist 0.0095, rate ~0.0598 → score 4.39. Built from a large ffmpeg
  parameter grid search (grid search results are published in the repo).
- Submission format: a public PR containing
  1. download link to `archive.zip` (the compressed data),
  2. `inflate.sh` — bash script that reconstructs raw video frames from `archive/`.
  Compression script optional. Final ranking = public leaderboard only.
- Evaluation constraints:
  - 30-minute time limit for the official evaluation (decode side).
  - Runtime choices: GitHub `linux-nvidia-t4` (16GB VRAM, 26GB RAM) or
    `ubuntu-latest` CPU (4 CPU, 16GB RAM).
  - External libraries/tools are free, BUT large artifacts (neural net weights,
    meshes, etc.) must go inside the archive and count toward compressed size.
  - Asymmetry to exploit: encode (training) happens offline on my hardware with
    no time limit; only decode is time-limited → favors per-video overfitting (INR).
- Dev resources: `test_videos.zip` (2.4 GB, 64 comma2k19 driving videos),
  `evaluate.sh`, scalable dataloader.
- Challenge is still open; submitting is an explicit path to job/internship interview.
- Prior winners: 1st @SajayR (PR #101); honorary open-code prizes @Quantizr (#55),
  @AaronLeslie138 (#95 + best write-up), @valtterivalo (#105). Their PRs on the
  commaai repo contain scores and often code — worth reading locally.

## Area background (neural video compression, one pass)

1. **Classical codecs** (H.264/HEVC/AV1 via ffmpeg): motion compensation +
   transform + entropy coding. The tuned-ffmpeg baseline is the first bar to clear.
2. **Autoencoder learned codecs** (DCVC series): trained encoder/decoder + learned
   entropy model, rate–distortion optimized end to end. Research SOTA but heavy;
   decoder weights would count against archive size here.
3. **Implicit neural representations (INR)** — family of the SOTA submission:
   overfit a small network to ONE video; the compressed file is the pruned +
   quantized + entropy-coded weights. NeRV → HNeRV → HiNeRV lineage.
   - HNeRV = "hybrid": tiny encoder makes a small content-adaptive per-frame
     embedding; decoder upsamples it. Much faster training / better quality than
     pure index→frame NeRV.
   - HiNeRV: hierarchical positional encodings + refined train/prune/quantize
     pipeline; ~72% BD-rate gain over HNeRV.
   - Analogy to my ACE background: ACE overfits a scene-coordinate MLP per scene;
     HNeRV overfits a decoder per video. Same per-instance representation philosophy.

## Why the metric is the whole game (task-aware compression)

- Distortion is measured through comma's SegNet + PoseNet → optimize "does the
  driving stack still see the same thing," not pixel fidelity.
- Consequences:
  - Sky / hood / fine texture can be destroyed nearly for free.
  - Road edges, lane lines, vanishing-point geometry are precious (PoseNet reads
    ego-motion from them).
  - The eval networks are provided → can backprop THROUGH them: use
    SegNet/PoseNet feature loss as the training distortion term. Biggest lever.
- Driving footage is highly self-similar (road bottom, sky top, forward motion)
  — exactly what per-video INRs exploit.

## SOTA submission decode: `rhnerv_latent_polish` (a12dongithub)

Name breakdown + repo description ("score-aware sparse-encoder, predictor with
refusal modes, meta-Lagrangian search engine", tagged `hnerv`):

- **rHNeRV**: HNeRV variant, "r" likely residual — predict frame t from previous
  content + learned residual (neural analog of I-frames vs P-frames).
- **Score-aware**: trains against the actual challenge metric (SegNet/PoseNet
  distortion), not MSE.
- **Predictor with refusal modes**: temporal predictor can "refuse" and fall back
  to explicit storage when prediction fails (scene changes, large motion).
- **Meta-Lagrangian search**: per-video search over λ in `min D + λR` plus
  hyperparameters (width, quant bits, epochs). Score sums linearly over videos →
  per-video operating points are exactly right.
- **latent_polish**: freeze (most of) the trained decoder, then directly
  gradient-optimize the per-frame latent embeddings against reconstruction/task
  loss — better values for bits already in the payload, so ~free quality.
  Likely includes a post-quantization polish pass and possibly polishing through
  the eval networks. Structurally similar to ACE-style per-scene refinement.
- Plausible full pipeline: per-video HNeRV model → task-aware loss training →
  prune/quantize → entropy-code weights+latents → latent polish post-quantization
  → per-video Lagrangian search → zip + `inflate.sh` that decodes on T4 in <30min.
- STATUS: I have NOT yet read the actual submission README/code (GitHub blocked
  automated fetch in the chat session). First Claude Code task: clone and read it,
  verify/correct this reconstruction.

## My attack plan

1. Clone commaai repo, run `evaluate.sh` on `baseline_fast`, read the eval code —
   understand exactly how PoseNet/SegNet distortion is computed (feature space,
   preprocessing, resolution). The metric is the spec.
2. Reproduce/understand the ffmpeg grid-search frontier on this metric. Cheap
   strong idea to test: ffmpeg + tiny shared learned decode-side restoration net.
3. INR route: vanilla HNeRV overfit on one comma clip → swap MSE for task-aware
   (SegNet/PoseNet feature) loss → quantization-aware training → latent polish.
   Read HNeRV then HiNeRV papers.
4. Read winning PRs (#101, #55, #95, #105) and AaronLeslie138's write-up.
5. Clone a12dongithub's fork, branch `submission/rhnerv_latent_polish`, study
   `submissions/rhnerv_latent_polish/` and their codec library; decide what to
   reuse vs. where to beat it.

## Setup commands (from challenge README)

```bash
git clone https://github.com/commaai/comma_video_compression_challenge.git
cd comma_video_compression_challenge
sudo apt-get update && sudo apt-get install -y git-lfs ffmpeg   # or brew on macOS
git lfs install && git lfs pull
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --group cpu    # cpu | cu126 | cu128 | cu130 | mps
source .venv/bin/activate
bash evaluate.sh --submission-dir ./submissions/baseline_fast --device cpu  # cpu|cuda|mps

# SOTA fork
git clone https://github.com/a12dongithub/comma_video_compression_challenge.git a12don_fork
cd a12don_fork && git checkout submission/rhnerv_latent_polish
```
