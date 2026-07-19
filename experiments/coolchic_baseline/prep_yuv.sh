#!/usr/bin/env bash
# Decode videos/0.mkv to a raw YUV420 file for Cool-Chic.
# The source is already yuv420p HEVC, so this is a lossless decode, no
# colorspace conversion. Filename convention <name>_<W>x<H>_<fps>p_yuv420_8b.yuv
# is REQUIRED: Cool-Chic parses geometry from the second underscore token.
#
# Usage: bash prep_yuv.sh [N_FRAMES]   (default 200)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
N_FRAMES="${1:-200}"
OUT_DIR="$HERE/data"
mkdir -p "$OUT_DIR"
# Padded to 1168x880: Cool-Chic's RAFT motion init needs H,W divisible by 8
# (874x1164 is not). Padding is cropped back off in score_coolchic.py.
OUT="$OUT_DIR/video0_1168x880_20p_yuv420_8b.yuv"

ffmpeg -y -loglevel error -i "$ROOT/videos/0.mkv" -frames:v "$N_FRAMES" \
  -vf "pad=1168:880:0:0" -pix_fmt yuv420p -f rawvideo "$OUT"

BYTES=$(stat -c%s "$OUT")
FRAME_BYTES=$((1168 * 880 * 3 / 2))
echo "wrote $OUT"
echo "  $BYTES bytes = $((BYTES / FRAME_BYTES)) frames (expected $N_FRAMES)"
