#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Build the first environment package for the clean compact release.
"${PYTHON_BIN}" "${ROOT_DIR}/mini_benchmark/build_clean_ship_core_lite_environment.py" \
  --clean-root "${ROOT_DIR}/mini_benchmark/clean_ship_core_lite_v1" \
  --stage10-dir "${ROOT_DIR}/data_interim/10_minlen_filtered/partitions" \
  --output-dir "${ROOT_DIR}/mini_benchmark/clean_ship_core_lite_v1/environment_v1"
