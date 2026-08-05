#!/usr/bin/env bash
# Challenge harness entry point: reconstruct raw frames from archive/.
# Usage: inflate.sh <archive_dir> <inflated_dir> <video_names_file>
#
# Requires the Cool-Chic 5.0.1 stack. Set COOLCHIC_REPO to the checkout
# (default ~/Cool-Chic). See README.md for the environment.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARCHIVE_DIR="${1:?archive dir}"
INFLATED_DIR="${2:?inflated dir}"
NAMES_FILE="${3:?video names file}"

# Prefer the conda env's libstdc++ (Cool-Chic's numpy needs a recent GLIBCXX).
if [ -n "${CONDA_PREFIX:-}" ] && [ -e "$CONDA_PREFIX/lib/libstdc++.so.6" ]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

python "$HERE/inflate.py" "$ARCHIVE_DIR" "$INFLATED_DIR" "$NAMES_FILE"
