#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Rebuild archive.zip byte-for-byte from the raw encoder inputs in encoder/.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -n "${PACT_PYTHON_BIN:-}" ]; then
  PYTHON_BIN="$PACT_PYTHON_BIN"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN=python
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
else
  echo "ERROR: neither python nor python3 is available" >&2
  exit 127
fi

"$PYTHON_BIN" "$HERE/compress.py"
