#!/usr/bin/env bash
# Decode videos/0.mkv to a raw YUV420 file for Cool-Chic.
# Filename convention <name>_<W>x<H>_<fps>p_yuv420_8b.yuv is REQUIRED:
# Cool-Chic parses geometry from the second underscore token.
#
# Usage: bash prep_yuv.sh [N_FRAMES] [RES]
#   N_FRAMES: default 200
#   RES: "full" (default; lossless decode padded to 1168x880 — Cool-Chic's
#        RAFT motion init needs H,W divisible by 8, and 874x1164 is not;
#        padding is cropped back off in score_coolchic.py)
#        "eval" (512x384, the judges' resolution; bilinear downscale matching
#        the metric's own resize; divisible by 8, no padding needed)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
N_FRAMES="${1:-200}"
RES="${2:-full}"
OUT_DIR="$HERE/data"
mkdir -p "$OUT_DIR"

if [ "$RES" = "eval" ]; then
  OUT="$OUT_DIR/video0_512x384_20p_yuv420_8b.yuv"
  VF="scale=512:384:flags=bilinear"
  W=512; H=384
else
  OUT="$OUT_DIR/video0_1168x880_20p_yuv420_8b.yuv"
  VF="pad=1168:880:0:0"
  W=1168; H=880
fi

ffmpeg -y -loglevel error -i "$ROOT/videos/0.mkv" -frames:v "$N_FRAMES" \
  -vf "$VF" -pix_fmt yuv420p -f rawvideo "$OUT"

BYTES=$(stat -c%s "$OUT")
FRAME_BYTES=$((W * H * 3 / 2))
echo "wrote $OUT"
echo "  $BYTES bytes = $((BYTES / FRAME_BYTES)) frames (expected $N_FRAMES)"
