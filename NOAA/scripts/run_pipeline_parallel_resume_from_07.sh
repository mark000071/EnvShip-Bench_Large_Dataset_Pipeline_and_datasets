#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${ROOT_DIR}/configs/dataset_v1.yaml"
PYTHON_BIN="${PYTHON_BIN:-/mnt/nfs/kun/conda_envs/deepjscc/bin/python}"
JOBS="${JOBS:-8}"

find "${ROOT_DIR}/data_interim/06_underway_only/partitions" -maxdepth 1 -type f -name 'part-*.csv.gz' | sort | \
  xargs -I{} -P "${JOBS}" \
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/07_interpolate_short_gaps.py" --input "{}" --output "${ROOT_DIR}/data_interim/07_gap_imputed" --config "${CONFIG}"

find "${ROOT_DIR}/data_interim/07_gap_imputed/partitions" -maxdepth 1 -type f -name 'part-*.csv.gz' | sort | \
  xargs -I{} -P "${JOBS}" \
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/08_resample_20s.py" --input "{}" --output "${ROOT_DIR}/data_interim/08_resampled_20s" --config "${CONFIG}"

find "${ROOT_DIR}/data_interim/08_resampled_20s/partitions" -maxdepth 1 -type f -name 'part-*.csv.gz' | sort | \
  xargs -I{} -P "${JOBS}" \
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/09_second_pass_anomaly_check.py" --input "{}" --output "${ROOT_DIR}/data_interim/09_rechecked" --config "${CONFIG}"

find "${ROOT_DIR}/data_interim/09_rechecked/partitions" -maxdepth 1 -type f -name 'part-*.csv.gz' | sort | \
  xargs -I{} -P "${JOBS}" \
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/10_filter_short_segments.py" --input "{}" --output "${ROOT_DIR}/data_interim/10_minlen_filtered" --config "${CONFIG}"

find "${ROOT_DIR}/data_interim/10_minlen_filtered/partitions" -maxdepth 1 -type f -name 'part-*.csv.gz' | sort | \
  xargs -I{} -P "${JOBS}" \
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/11_make_sliding_windows.py" --input "{}" --output "${ROOT_DIR}/data_interim/11_windows" --config "${CONFIG}"

find "${ROOT_DIR}/data_interim/11_windows/partitions" -maxdepth 1 -type f -name 'part-*.csv.gz' | sort | \
  xargs -I{} -P "${JOBS}" \
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/12_compute_quality_labels.py" --input "{}" --output "${ROOT_DIR}/data_interim/12_quality_labeled" --config "${CONFIG}"

find "${ROOT_DIR}/data_interim/12_quality_labeled/partitions" -maxdepth 1 -type f -name 'part-*.csv.gz' | sort | \
  xargs -I{} -P "${JOBS}" \
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/13_export_benchmark.py" --input "{}" --output "${ROOT_DIR}/benchmark" --config "${CONFIG}"
