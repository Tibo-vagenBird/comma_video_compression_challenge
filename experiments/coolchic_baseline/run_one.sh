#!/usr/bin/env bash
# Encode + decode one Cool-Chic rate point on the prepped YUV.
# Run inside the *coolchic* conda env (torch>=2.11), NOT comma-compress.
#
# Usage: bash run_one.sh <coolchic_repo_path> <lambda> [n_frames] [gpu]
#   e.g. bash run_one.sh ~/Cool-Chic 0.02 200 3
#
# Default coding structure: frame 0 intra, all remaining frames hierarchical
# B-frames (Cool-Chic's default random-access GOP). Higher lambda = fewer bits.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CCREPO="${1:?arg1: path to Cool-Chic repo}"
LMBDA="${2:?arg2: lambda, e.g. 0.02}"
N_FRAMES="${3:-200}"
GPU="${4:-0}"

YUV="$HERE/data/video0_1168x880_20p_yuv420_8b.yuv"
[ -f "$YUV" ] || { echo "ERROR: $YUV missing — run prep_yuv.sh first" >&2; exit 1; }

RUN="$HERE/runs/lmbda_${LMBDA}_n${N_FRAMES}"
mkdir -p "$RUN/work"

# Live log lines instead of block-buffered chunks through tee
export PYTHONUNBUFFERED=1

# Refuse to run on CPU: a wrong GPU index makes CUDA invisible and torch
# silently falls back to CPU (hours per frame instead of minutes).
CUDA_VISIBLE_DEVICES="$GPU" python - <<'EOF' || { echo "ERROR: GPU not visible with CUDA_VISIBLE_DEVICES=$GPU — check 'nvidia-smi -L' and pass a valid index" >&2; exit 1; }
import sys, torch
ok = torch.cuda.is_available()
print(f"cuda available: {ok} | visible devices: {torch.cuda.device_count()}")
sys.exit(0 if ok else 1)
EOF

echo "=== encode: lambda=$LMBDA n_frames=$N_FRAMES gpu=$GPU ==="
CUDA_VISIBLE_DEVICES="$GPU" python "$HERE/encode_video.py" \
  --coolchic "$CCREPO" \
  -i "$YUV" \
  -o "$RUN/bitstream.cool" \
  --workdir "$RUN/work" \
  --intra_pos 0 \
  --n_frames "$N_FRAMES" \
  --lmbda "$LMBDA" 2>&1 | tee "$RUN/encode.log"

cd "$CCREPO"

BYTES=$(stat -c%s "$RUN/bitstream.cool")
echo "=== bitstream: $BYTES bytes for $N_FRAMES frames ==="

echo "=== decode ==="
CUDA_VISIBLE_DEVICES="$GPU" python cc_decode.py \
  -i "$RUN/bitstream.cool" \
  -o "$RUN/recon_1168x880_20p_yuv420_8b.yuv" 2>&1 | tee "$RUN/decode.log"

echo "done: $RUN"
echo "next (comma-compress env, from challenge repo root):"
echo "  python experiments/coolchic_baseline/score_coolchic.py --run $RUN"
