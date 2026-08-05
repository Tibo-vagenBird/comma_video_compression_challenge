#!/usr/bin/env bash
# Encode + decode one Cool-Chic rate point on the prepped YUV.
# Run inside the *coolchic* conda env (torch>=2.11), NOT comma-compress.
#
# Usage: bash run_one.sh <coolchic_repo> <lambda> [n_frames] [gpu] [struct] [res] [tune] [run_tag]
#   n_frames: default 200
#   gpu:      GPU index (default 0). "auto" picks the first idle GPU.
#   struct:   "gop16" (default; hierarchical B, P every 16) or "ippp"
#             (low-delay, every frame P referencing its predecessor — better
#             within-pair temporal consistency, which PoseNet measures)
#   res:      "full" (default; 1168x880 padded) or "eval" (512x384, judges' res)
#   tune:     "mse" (default) or "comma" (SegNet/PoseNet judge losses; needs
#             targets from prep_targets.py and timm/smp/einops/safetensors
#             installed in the coolchic env)
#   run_tag:  optional. Empty = auto-increment a fresh _rNN folder each run
#             (clean run, keeps history, no rm needed). Pass an explicit tag
#             (e.g. r03) to reuse/resume that exact folder.
#
# Examples:
#   bash run_one.sh ~/Cool-Chic 40 32 auto ippp eval comma       # fresh _rNN
#   bash run_one.sh ~/Cool-Chic 40 32 auto ippp eval comma r03   # reuse r03
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
CCREPO="${1:?arg1: path to Cool-Chic repo}"
LMBDA="${2:?arg2: lambda, e.g. 0.02}"
N_FRAMES="${3:-200}"
GPU="${4:-0}"
STRUCT="${5:-gop16}"
RES="${6:-full}"
TUNE="${7:-mse}"
RUN_TAG="${8:-}"

# ---------------------------------------------------------------------------
# Environment hardening
# ---------------------------------------------------------------------------
# 1) Prefer the conda env's libstdc++ over the (older) system one. pip wheels
#    (numpy 2.4+) need GLIBCXX_3.4.29 which old Ubuntu system libs lack. Must
#    run BEFORE any python invocation. Harmless if the file isn't there.
if [ -n "${CONDA_PREFIX:-}" ] && [ -e "$CONDA_PREFIX/lib/libstdc++.so.6" ]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi
# 2) Live log lines instead of block-buffered chunks through tee.
export PYTHONUNBUFFERED=1

# ---------------------------------------------------------------------------
# GPU selection
# ---------------------------------------------------------------------------
if [ "$GPU" = "auto" ]; then
  # Pick the GPU with the most free memory.
  GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | sort -t',' -k2 -n -r | head -1 | cut -d',' -f1 | tr -d ' ')
  echo "auto-selected GPU $GPU (most free memory)"
fi

# Refuse to run on CPU: a wrong GPU index makes CUDA invisible and torch
# silently falls back to CPU (hours per frame instead of minutes).
CUDA_VISIBLE_DEVICES="$GPU" python - <<'EOF' || { echo "ERROR: GPU not visible with CUDA_VISIBLE_DEVICES=$GPU — check 'nvidia-smi -L' and pass a valid index" >&2; exit 1; }
import sys, torch
ok = torch.cuda.is_available()
print(f"cuda available: {ok} | visible devices: {torch.cuda.device_count()}")
sys.exit(0 if ok else 1)
EOF

# ---------------------------------------------------------------------------
# Coding structure / resolution / tune
# ---------------------------------------------------------------------------
P_POS_ARG=""
STRUCT_TAG=""
if [ "$STRUCT" = "ippp" ]; then
  P_POS_ARG="--p_pos=1-$((N_FRAMES - 1))"
  STRUCT_TAG="_ippp"
fi

if [ "$RES" = "eval" ]; then
  GEOM="512x384"
  RES_TAG="_eval"
else
  GEOM="1168x880"
  RES_TAG=""
fi
YUV="$HERE/data/video0_${GEOM}_20p_yuv420_8b.yuv"
[ -f "$YUV" ] || { echo "ERROR: $YUV missing — run 'bash prep_yuv.sh $N_FRAMES $RES' first" >&2; exit 1; }

EXTRA_ARGS=""
TUNE_TAG=""
if [ "$TUNE" = "comma" ]; then
  TUNE_TAG="_comma"
  EXTRA_ARGS="--tune=comma"
  export COMMA_CHALLENGE_ROOT="$ROOT"
  export COMMA_TARGETS_PT="$HERE/data/targets_n${N_FRAMES}.pt"
  [ -f "$COMMA_TARGETS_PT" ] || { echo "ERROR: $COMMA_TARGETS_PT missing — run 'python experiments/coolchic_baseline/prep_targets.py --n-frames $N_FRAMES' (comma-compress env) first" >&2; exit 1; }
fi

RUN_BASE="lmbda_${LMBDA}_n${N_FRAMES}${STRUCT_TAG}${RES_TAG}${TUNE_TAG}"
if [ -n "$RUN_TAG" ]; then
  # Explicit tag: reuse/resume this exact folder.
  RUN="$HERE/runs/${RUN_BASE}_${RUN_TAG}"
else
  # Auto-increment: first unused _rNN, so every run is clean and history is kept.
  idx=1
  while [ -e "$HERE/runs/${RUN_BASE}_r$(printf '%02d' "$idx")" ]; do
    idx=$((idx + 1))
  done
  RUN="$HERE/runs/${RUN_BASE}_r$(printf '%02d' "$idx")"
fi
mkdir -p "$RUN/work"
echo "run dir: $RUN"

# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------
echo "=== encode: lambda=$LMBDA n_frames=$N_FRAMES gpu=$GPU struct=$STRUCT res=$RES tune=$TUNE ==="
CUDA_VISIBLE_DEVICES="$GPU" python "$HERE/encode_video.py" \
  --coolchic "$CCREPO" \
  -i "$YUV" \
  -o "$RUN/bitstream.cool" \
  --workdir "$RUN/work" \
  --intra_pos 0 \
  --n_frames "$N_FRAMES" \
  $P_POS_ARG \
  --extra_args="$EXTRA_ARGS" \
  --lmbda "$LMBDA" 2>&1 | tee "$RUN/encode.log"

BYTES=$(stat -c%s "$RUN/bitstream.cool")
echo "=== bitstream: $BYTES bytes for $N_FRAMES frames ==="

# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------
cd "$CCREPO"
echo "=== decode ==="
CUDA_VISIBLE_DEVICES="$GPU" python cc_decode.py \
  -i "$RUN/bitstream.cool" \
  -o "$RUN/recon_${GEOM}_20p_yuv420_8b.yuv" 2>&1 | tee "$RUN/decode.log"

echo "done: $RUN"
echo "next (comma-compress env, from challenge repo root):"
echo "  python experiments/coolchic_baseline/score_coolchic.py --run $RUN"
