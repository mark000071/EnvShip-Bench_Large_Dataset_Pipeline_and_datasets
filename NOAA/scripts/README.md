# Scripts

Scripts are numbered to match the benchmark protocol order and can be executed stage-by-stage.

## Core stages

- `01_standardize_fields.py`
  Standardize DMA raw fields, timestamps, units, and AIS sentinel values.
- `02_basic_filter.py`
  Apply hard legality filtering for MMSI, time, coordinates, and navigation fields.
- `03_sort_dedup.py`
  Partition by MMSI, sort, and deduplicate same-timestamp records.
- `04_shiptype_speed_filter.py`
  Apply global and ship-type-dependent speed thresholds.
- `05_segment_tracks.py`
  Build continuous track segments using time gaps and implied-speed checks.
- `06_remove_anchorage.py`
  Remove low-motion and anchoring/mooring segments for the main benchmark.
- `07_interpolate_short_gaps.py`
  Interpolate only short gaps up to 120 seconds.
- `08_resample_20s.py`
  Resample valid segments to a global 20-second time grid.
- `09_second_pass_anomaly_check.py`
  Remove residual drift spikes after resampling.
- `10_filter_short_segments.py`
  Keep only sufficiently long segments.
- `11_make_sliding_windows.py`
  Build fixed `30 history + 30 future` windows.
- `12_compute_quality_labels.py`
  Compute interpolation ratios, displacements, and `core/full/drop` labels.
- `13_export_benchmark.py`
  Export final split-wise benchmark shards.
- `14_collect_partition_summaries.py`
  Merge per-partition summary files into stage-level summaries.

## Shared utilities

- `pipeline_utils.py`
  Common readers, writers, partition helpers, coordinate transforms, and config parsing.

## Pipeline runners

- `run_pipeline.sh`
  Simple sequential skeleton runner.
- `run_pipeline_parallel_dma_2025_09.sh`
  Parallel monthly runner for DMA `2025-09`.
- `run_pipeline_resume_from_05.sh`
  Resume runner from stage `05`.
- `run_pipeline_parallel_resume_from_07.sh`
  Parallel resume runner from stage `07`.
- `run_one_day_pilot_2025_09_01.sh`
  One-day pilot runner for validating the end-to-end pipeline before scaling up.
