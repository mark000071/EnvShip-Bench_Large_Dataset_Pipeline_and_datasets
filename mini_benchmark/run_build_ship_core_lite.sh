#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" "${ROOT_DIR}/mini_benchmark/build_ship_core_lite.py" \
  --benchmark-root "${ROOT_DIR}/benchmark/core" \
  --output-root "${ROOT_DIR}/mini_benchmark/ship_core_lite" \
  --train-target 32000 \
  --val-target 4000 \
  --test-target 4000
