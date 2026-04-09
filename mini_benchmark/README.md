# Mini Benchmarks

This directory contains the compact releases used for fast iteration, ablation studies, and reproducible model development.

## Ship-Core-Lite

`Ship-Core-Lite` is a representative subset of the full DMA `benchmark/core` release. It keeps the same prediction protocol and sample format while reducing the total volume to something that is practical for routine iteration.

The subset follows the same forecasting setup as the full benchmark:

- observation horizon: 10 minutes
- prediction horizon: 10 minutes
- sampling interval: 20 seconds
- history length: 30 points
- future length: 30 points

Released size:

- train: 32,000
- val: 4,000
- test: 4,000

### How it is built

`Ship-Core-Lite` is not a random slice of the full benchmark. The builder uses stratified sampling to preserve representative motion patterns while limiting heavy redundancy from overlapping windows:

1. samples are grouped by vessel category, speed range, and future displacement range
2. within each coarse group, turning behavior is used to retain directional diversity
3. split quotas are assigned with `sqrt(count)` weighting to avoid letting dominant traffic modes overwhelm the subset
4. repeated windows from the same vessel and segment are capped explicitly

The result is a compact benchmark that stays close to the full monthly release in format and motion profile.

## clean_ship_core_lite_v1

`clean_ship_core_lite_v1` is a stricter compact subset built for quality-first experiments. It uses additional motion-quality screening, narrower vessel-type selection, and stronger redundancy control. The release includes:

- compact train/val/test splits
- exact sample ID lists
- split-level reports and selected-metadata tables
- environment context packages (`environment_v1`, `environment_v2`) in the full data release
- a synchronized social context package (`social_env_v1`) in the full data release

## Main files

- `build_ship_core_lite.py`
  Builds the representative compact subset from `benchmark/core`.
- `build_clean_ship_core_lite.py`
  Builds the stricter quality-first subset.
- `build_clean_ship_core_lite_environment.py`
  Constructs the first environment context package for the clean subset.
- `build_clean_ship_core_lite_environment_v2.py`
  Builds the second environment context package with richer descriptors.
- `run_build_ship_core_lite.sh`
  Convenience runner for rebuilding `Ship-Core-Lite`.

## Layout

```text
mini_benchmark/
├── README.md
├── build_ship_core_lite.py
├── build_clean_ship_core_lite.py
├── build_clean_ship_core_lite_environment.py
├── build_clean_ship_core_lite_environment_v2.py
├── run_build_ship_core_lite.sh
├── ship_core_lite/
└── clean_ship_core_lite_v1/
```

## Compatibility

Both compact releases preserve the row-wise sample format of the full benchmark. In practice, code that reads `benchmark/core` can switch to a compact version by changing only the dataset root.

## Rebuilding

To rebuild `Ship-Core-Lite` locally:

```bash
cd mini_benchmark
bash run_build_ship_core_lite.sh
```

The released outputs include sample lists and reports so the subset can be audited and regenerated deterministically.
