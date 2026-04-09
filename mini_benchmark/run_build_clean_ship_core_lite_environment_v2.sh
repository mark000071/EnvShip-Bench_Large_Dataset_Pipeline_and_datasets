#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" "${ROOT_DIR}/mini_benchmark/build_clean_ship_core_lite_environment_v2.py" \
  --clean-root "${ROOT_DIR}/mini_benchmark/clean_ship_core_lite_v1" \
  --output-dir "${ROOT_DIR}/mini_benchmark/clean_ship_core_lite_v1/environment_v2" \
  --source-env-root "${ROOT_DIR}/mini_benchmark/clean_ship_core_lite_v1/environment_v1"
