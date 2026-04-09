#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${ROOT_DIR}/configs/dataset_v1.yaml"
PYTHON_BIN="${PYTHON_BIN:-/mnt/nfs/kun/conda_envs/deepjscc/bin/python}"
RAW_FILE="${ROOT_DIR}/data_raw/dma/incoming/2025-09/aisdk-2025-09-01.zip"
PILOT_ROOT="${ROOT_DIR}/tmp/pilot_2025_09_01"

rm -rf "${PILOT_ROOT}"
mkdir -p "${PILOT_ROOT}"/{01,02,03,04,05,06,07,08,09,10,11,12,benchmark}

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/01_standardize_fields.py" --input "${RAW_FILE}" --output "${PILOT_ROOT}/01" --config "${CONFIG}"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/02_basic_filter.py" --input "${PILOT_ROOT}/01/aisdk-2025-09-01.csv.gz" --output "${PILOT_ROOT}/02" --config "${CONFIG}"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/03_sort_dedup.py" --input "${PILOT_ROOT}/02" --output "${PILOT_ROOT}/03" --config "${CONFIG}"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/04_shiptype_speed_filter.py" --input "${PILOT_ROOT}/03" --output "${PILOT_ROOT}/04" --config "${CONFIG}"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/05_segment_tracks.py" --input "${PILOT_ROOT}/04" --output "${PILOT_ROOT}/05" --config "${CONFIG}"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/06_remove_anchorage.py" --input "${PILOT_ROOT}/05" --output "${PILOT_ROOT}/06" --config "${CONFIG}"
PYTHON_BIN="${PYTHON_BIN}" JOBS=8 ROOT_DIR="${PILOT_ROOT}" CONFIG="${CONFIG}" bash -lc '
find "$ROOT_DIR/06/partitions" -maxdepth 1 -type f -name "part-*.csv.gz" | sort | xargs -I{} -P "$JOBS" "$PYTHON_BIN" "/mnt/nfs/kun/DeepJSCC/ship_trajectory_datesets/scripts/07_interpolate_short_gaps.py" --input "{}" --output "$ROOT_DIR/07" --config "$CONFIG"
find "$ROOT_DIR/07/partitions" -maxdepth 1 -type f -name "part-*.csv.gz" | sort | xargs -I{} -P "$JOBS" "$PYTHON_BIN" "/mnt/nfs/kun/DeepJSCC/ship_trajectory_datesets/scripts/08_resample_20s.py" --input "{}" --output "$ROOT_DIR/08" --config "$CONFIG"
find "$ROOT_DIR/08/partitions" -maxdepth 1 -type f -name "part-*.csv.gz" | sort | xargs -I{} -P "$JOBS" "$PYTHON_BIN" "/mnt/nfs/kun/DeepJSCC/ship_trajectory_datesets/scripts/09_second_pass_anomaly_check.py" --input "{}" --output "$ROOT_DIR/09" --config "$CONFIG"
find "$ROOT_DIR/09/partitions" -maxdepth 1 -type f -name "part-*.csv.gz" | sort | xargs -I{} -P "$JOBS" "$PYTHON_BIN" "/mnt/nfs/kun/DeepJSCC/ship_trajectory_datesets/scripts/10_filter_short_segments.py" --input "{}" --output "$ROOT_DIR/10" --config "$CONFIG"
find "$ROOT_DIR/10/partitions" -maxdepth 1 -type f -name "part-*.csv.gz" | sort | xargs -I{} -P "$JOBS" "$PYTHON_BIN" "/mnt/nfs/kun/DeepJSCC/ship_trajectory_datesets/scripts/11_make_sliding_windows.py" --input "{}" --output "$ROOT_DIR/11" --config "$CONFIG"
find "$ROOT_DIR/11/partitions" -maxdepth 1 -type f -name "part-*.csv.gz" | sort | xargs -I{} -P "$JOBS" "$PYTHON_BIN" "/mnt/nfs/kun/DeepJSCC/ship_trajectory_datesets/scripts/12_compute_quality_labels.py" --input "{}" --output "$ROOT_DIR/12" --config "$CONFIG"
find "$ROOT_DIR/12/partitions" -maxdepth 1 -type f -name "part-*.csv.gz" | sort | xargs -I{} -P "$JOBS" "$PYTHON_BIN" "/mnt/nfs/kun/DeepJSCC/ship_trajectory_datesets/scripts/13_export_benchmark.py" --input "{}" --output "$ROOT_DIR/benchmark" --config "$CONFIG"
'
