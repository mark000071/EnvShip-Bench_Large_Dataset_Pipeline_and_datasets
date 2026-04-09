#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Rebuild the released clean subset from the exported DMA core benchmark.
"${PYTHON_BIN}" "${ROOT_DIR}/mini_benchmark/build_clean_ship_core_lite.py" \
  --benchmark-root "${ROOT_DIR}/benchmark/core" \
  --output-root "${ROOT_DIR}/mini_benchmark/clean_ship_core_lite_v1" \
  --train-target 19500 \
  --val-target 2400 \
  --test-target 2100
