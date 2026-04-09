#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/nfs/kun/DeepJSCC/NOAA_ship_trajectory_datasets"
PYTHON="/mnt/nfs/kun/conda_envs/deepjscc/bin/python"
MINI_BENCH_DIR="$ROOT/mini_bench"
CLEAN_ROOT="$MINI_BENCH_DIR/clean_ship_core_lite_v1"

cd "$ROOT"

if [[ ! -f "$CLEAN_ROOT/summary.json" ]]; then
  stdbuf -oL -eL "$PYTHON" "$MINI_BENCH_DIR/build_clean_ship_core_lite.py"
fi

if [[ ! -f "$CLEAN_ROOT/environment_v1/summary.json" ]]; then
  stdbuf -oL -eL "$PYTHON" \
    "$MINI_BENCH_DIR/build_clean_ship_core_lite_environment.py" \
    --tile-deg 1.0 \
    --sleep-between-queries 0.0
fi

if [[ ! -f "$CLEAN_ROOT/environment_v1/scene_groups_v1/summary.json" ]]; then
  stdbuf -oL -eL "$PYTHON" "$MINI_BENCH_DIR/build_clean_ship_scene_groups.py"
fi

if [[ ! -f "$CLEAN_ROOT/social_env_v1/summary.json" ]]; then
  stdbuf -oL -eL "$PYTHON" "$MINI_BENCH_DIR/build_clean_ship_core_lite_social_env.py"
fi

echo "[done] NOAA clean_ship_core_lite_v1 pipeline finished"
