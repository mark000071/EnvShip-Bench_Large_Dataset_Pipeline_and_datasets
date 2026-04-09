#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${ROOT_DIR}/configs/dataset_v1.yaml"
RAW_DIR="${ROOT_DIR}/data_raw/dma/incoming/2025-09"
JOBS="${JOBS:-8}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/nfs/kun/conda_envs/deepjscc/bin/python}"

# Stages 01 and 02 are parallelized per daily raw archive. The remaining
# stages operate on partitioned monthly outputs.
find "${RAW_DIR}" -maxdepth 1 -type f -name 'aisdk-2025-09-*.zip' | sort | \
  xargs -I{} -P "${JOBS}" \
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/01_standardize_fields.py" --input "{}" --output "${ROOT_DIR}/data_interim/01_standardized" --config "${CONFIG}"

find "${ROOT_DIR}/data_interim/01_standardized" -maxdepth 1 -type f -name 'aisdk-2025-09-*.csv.gz' | sort | \
  xargs -I{} -P "${JOBS}" \
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/02_basic_filter.py" --input "{}" --output "${ROOT_DIR}/data_interim/02_filtered" --config "${CONFIG}"

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/03_sort_dedup.py" --input "${ROOT_DIR}/data_interim/02_filtered" --output "${ROOT_DIR}/data_interim/03_deduped" --config "${CONFIG}"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/04_shiptype_speed_filter.py" --input "${ROOT_DIR}/data_interim/03_deduped" --output "${ROOT_DIR}/data_interim/04_shiptype_speed_filtered" --config "${CONFIG}"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/05_segment_tracks.py" --input "${ROOT_DIR}/data_interim/04_shiptype_speed_filtered" --output "${ROOT_DIR}/data_interim/05_segmented" --config "${CONFIG}"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/06_remove_anchorage.py" --input "${ROOT_DIR}/data_interim/05_segmented" --output "${ROOT_DIR}/data_interim/06_underway_only" --config "${CONFIG}"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/07_interpolate_short_gaps.py" --input "${ROOT_DIR}/data_interim/06_underway_only" --output "${ROOT_DIR}/data_interim/07_gap_imputed" --config "${CONFIG}"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/08_resample_20s.py" --input "${ROOT_DIR}/data_interim/07_gap_imputed" --output "${ROOT_DIR}/data_interim/08_resampled_20s" --config "${CONFIG}"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/09_second_pass_anomaly_check.py" --input "${ROOT_DIR}/data_interim/08_resampled_20s" --output "${ROOT_DIR}/data_interim/09_rechecked" --config "${CONFIG}"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/10_filter_short_segments.py" --input "${ROOT_DIR}/data_interim/09_rechecked" --output "${ROOT_DIR}/data_interim/10_minlen_filtered" --config "${CONFIG}"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/11_make_sliding_windows.py" --input "${ROOT_DIR}/data_interim/10_minlen_filtered" --output "${ROOT_DIR}/data_interim/11_windows" --config "${CONFIG}"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/12_compute_quality_labels.py" --input "${ROOT_DIR}/data_interim/11_windows" --output "${ROOT_DIR}/data_interim/12_quality_labeled" --config "${CONFIG}"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/13_export_benchmark.py" --input "${ROOT_DIR}/data_interim/12_quality_labeled" --output "${ROOT_DIR}/benchmark" --config "${CONFIG}"
