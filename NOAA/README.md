# NOAA AIS March 2025 Benchmark Pipeline

This directory contains the NOAA-specific release notes, scripts, and compact metadata for the March 2025 benchmark branch.

## Overview

- Source: NOAA AIS daily raw files
- Month: `2025-03-01` to `2025-03-31`
- Task: short-horizon ship trajectory forecasting
- Observation window: `10 min`
- Prediction window: `10 min`
- Resample interval: `20 s`
- History points: `30`
- Future points: `30`
- Partition count: `64`

The pipeline is adapted from the DMA version in `ship_trajectory_datesets`, but the standardization layer was extended for NOAA raw schema:

- raw compression: `.csv.zst`
- timestamp field: `base_date_time`
- ship type field: `vessel_type`
- navigation status field: `status`

## Directory Layout

```text
NOAA/
├── benchmark/
│   ├── core/{train,val,test}/part-*.csv.gz
│   ├── full/{train,val,test}/part-*.csv.gz
│   └── part-*.csv.export_summary.json
├── benchmark_summary.json
├── configs/dataset_v1.yaml
├── data_raw/noaa/2025-03/ais-2025-03-*.csv.zst
├── data_interim/
│   ├── 01_standardized/
│   ├── 02_filtered/
│   ├── 03_deduped/
│   ├── 04_shiptype_speed_filtered/
│   ├── 05_segmented/
│   ├── 06_underway_only/
│   ├── 07_gap_imputed/
│   ├── 08_resampled_20s/
│   ├── 09_rechecked/
│   ├── 10_minlen_filtered/
│   ├── 11_windows/
│   └── 12_quality_labeled/
└── scripts/
    ├── download_noaa_2025_03.sh
    ├── run_pipeline_parallel_noaa_2025_03.sh
    └── run_pipeline_resume_from_08_noaa_2025_03.sh
```

The large NOAA benchmark shards and raw files are hosted on Hugging Face together with the DMA release.

## Pipeline Stages

1. `01_standardize_fields.py`
   Convert NOAA raw fields into the canonical schema and normalize sentinel values.
2. `02_basic_filter.py`
   Remove invalid MMSI, timestamp, latitude, longitude, and hard-invalid nav fields.
3. `03_sort_dedup.py`
   Sort by `mmsi + timestamp_utc` and remove duplicates.
4. `04_shiptype_speed_filter.py`
   Apply hard speed caps and ship-type-aware soft caps.
5. `05_segment_tracks.py`
   Split trajectories by large temporal gaps and implied-speed breaks.
6. `06_remove_anchorage.py`
   Remove anchoring, mooring, and low-motion segments.
7. `07_interpolate_short_gaps.py`
   Interpolate only short gaps.
8. `08_resample_20s.py`
   Resample valid trajectories to a 20-second grid.
9. `09_second_pass_anomaly_check.py`
   Remove residual post-resampling anomalies.
10. `10_filter_short_segments.py`
    Keep only segments long enough for `30 + 30` window slicing.
11. `11_make_sliding_windows.py`
    Create fixed-length forecast samples.
12. `12_compute_quality_labels.py`
    Compute interpolation ratios and assign `core/full/drop`.
13. `13_export_benchmark.py`
    Export final `core` and `full` releases into `train/val/test`.

## NOAA-Specific Scripts

- `scripts/download_noaa_2025_03.sh`
  Download all daily NOAA raw files for March 2025.
- `scripts/run_pipeline_parallel_noaa_2025_03.sh`
  Run the full NOAA monthly pipeline from raw data.
- `scripts/run_pipeline_resume_from_08_noaa_2025_03.sh`
  Resume from Stage 08 and continue to final benchmark export.

## Final Result Scale

Final benchmark counts:

- `core_train = 2,853,869`
- `core_val = 350,486`
- `core_test = 346,598`
- `full_train = 3,210,217`
- `full_val = 396,863`
- `full_test = 390,113`

Derived totals:

- `core_total = 3,550,953`
- `full_total = 3,997,193`
- `full_only_extra_over_core = 446,240`
- `drop_total = 16,461`

Sizes:

- raw NOAA March 2025 data: about `6.81 GB`
- final benchmark directory: about `5.1 GB`

## Completeness Check

The generated dataset was checked for structural completeness.

Verified:

- all `31` raw daily NOAA files exist
- `01_standardized` contains all `31` expected daily outputs
- `02_filtered` contains all `31` expected daily outputs
- stages `03` to `12` each contain all `64` expected partition outputs
- `benchmark/core/{train,val,test}` each contain all `64` shard files
- `benchmark/full/{train,val,test}` each contain all `64` shard files
- `benchmark` contains all `64` partition export summary files

Important note:

- The first run was interrupted during Stage `08_resample_20s`.
- The missing partitions were resumed successfully with `scripts/run_pipeline_resume_from_08_noaa_2025_03.sh`.
- Because of that interrupted first run, `data_interim/08_resampled_20s/summary.json` was not regenerated as a single global file.
- This does not indicate missing benchmark data: the `08` partition outputs are complete, and all downstream stages `09-13` completed successfully.

The consolidated audit is stored in `benchmark_summary.json`.

## Sample Format

Each row in `benchmark/{core,full}/{train,val,test}/part-*.csv.gz` is one sample.

Important columns:

- `sample_id`
- `mmsi`
- `segment_id`
- `hist_x_json`, `hist_y_json`
- `fut_x_json`, `fut_y_json`
- `hist_sog_json`
- `interp_ratio_total`
- `grid_interp_ratio_total`
- `quality_tier`
- `split`

## Re-running

Download raw data:

```bash
bash scripts/download_noaa_2025_03.sh
```

Run the full month:

```bash
JOBS=8 bash scripts/run_pipeline_parallel_noaa_2025_03.sh
```

Resume from Stage 08:

```bash
JOBS=4 bash scripts/run_pipeline_resume_from_08_noaa_2025_03.sh
```
