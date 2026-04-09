#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${ROOT_DIR}/data_raw/noaa/2025-03"
BASE_URL="https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2025"

mkdir -p "${RAW_DIR}"

for day in $(seq -w 1 31); do
  file="ais-2025-03-${day}.csv.zst"
  url="${BASE_URL}/${file}"
  out="${RAW_DIR}/${file}"
  echo "downloading ${file}"
  curl -L --fail --retry 3 -C - -o "${out}" "${url}"
done

