# Protocol v1

## Benchmark target

Build a high-quality AIS-based core dataset for short-horizon trajectory forecasting under one unified protocol.

## Fixed task definition

- Observation horizon: 10 minutes
- Prediction horizon: 10 minutes
- Resampling cadence: 20 seconds
- History points: 30
- Future points: 30
- Prediction target: future 30 positions
- Input fields: `lat`, `lon`, `sog`, `cog`, `heading`, `dt`, `ship_type`
- Main evaluation unit: meters

## Construction protocol

### 1. Field standardization and sentinel handling

Normalize source columns to a canonical schema and convert DMA sentinel values to `NaN` or explicit flags.

### 2. Basic legality filtering

Drop rows with invalid MMSI, impossible timestamps, invalid coordinates, or clearly corrupted motion fields.

### 3. Sort and deduplicate by MMSI

Sort by `mmsi`, `timestamp`; deduplicate exact and near-duplicate messages under a deterministic rule.

### 4. Ship-type speed-threshold filtering

Use ship-type-aware maximum plausible speeds to remove physically implausible records or segments.

### 5. Segment by 10 min gap and implied speed

Split trajectories when:

- consecutive messages are separated by more than 10 minutes, or
- implied transition speed exceeds a chosen threshold

### 6. Remove berthing / anchoring segments

Exclude stationary or near-stationary harbor / anchorage behavior from the forecasting benchmark.

### 7. Interpolate only short gaps `<= 120 s`

Interpolate only bounded missing spans that are short enough not to hide genuine route uncertainty.

### 8. Uniform 20 s resampling

Resample all valid underway segments to a fixed 20-second grid.

### 9. Second-pass anomaly detection

Recheck geometry and kinematics after resampling and interpolation.

### 10. Filter too-short segments

Keep only segments long enough to support at least one `10 min -> 10 min` sample under the fixed protocol.

### 11. Slice `10 -> 10` sliding windows

Generate training / validation / test samples with:

- 30 history steps
- 30 future steps
- 20-second cadence

### 12. Compute quality labels and interpolation ratio

Track sample-level metadata such as:

- interpolation ratio
- anchor / berth proximity flags
- motion smoothness
- anomaly counts
- region and ship-type tags

### 13. Export `Core` and `Full`

Release two versions:

- `Core`: strict quality subset for main leaderboard
- `Full`: broader coverage for large-scale training and ablation

## Canonical row schema

Suggested canonical columns for standardized messages:

- `mmsi`
- `timestamp`
- `lat`
- `lon`
- `sog`
- `cog`
- `heading`
- `ship_type`
- `nav_status`
- `source_file`
- `is_interpolated`

## Canonical sample schema

Suggested sample-level metadata:

- `sample_id`
- `segment_id`
- `mmsi`
- `region_id`
- `ship_type`
- `split`
- `hist_start_ts`
- `hist_end_ts`
- `pred_end_ts`
- `interp_ratio_hist`
- `interp_ratio_fut`
- `quality_tier`
- `version`
