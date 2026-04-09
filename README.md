# Ship-Env

Ship-Env is the public code release for *EnvShip-Bench: An Environment-Enhanced Benchmark for Short-Term Vessel Trajectory Prediction*. It contains the preprocessing pipeline used to turn raw AIS messages into benchmark-ready trajectory windows, together with compact releases, preview shards, visualization utilities, and paper support material.

Resources:

- Dataset: <https://huggingface.co/datasets/mark000071/EnvShip-Bench_An_Environment-Enhanced_Benchmark_for_Short-Term_Vessel_Trajectory_Prediction>
- Citation: `CITATION.cff`
- License: `LICENSE`

All released benchmarks follow one forecasting protocol:

- observation horizon: 10 minutes
- prediction horizon: 10 minutes
- sampling interval: 20 seconds
- history length: 30 points
- future length: 30 points

The full DMA and NOAA releases are hosted on Hugging Face:

- <https://huggingface.co/datasets/mark000071/EnvShip-Bench_An_Environment-Enhanced_Benchmark_for_Short-Term_Vessel_Trajectory_Prediction>

GitHub is used for code, documentation, manifests, and lightweight preview assets. Large benchmark shards stay on Hugging Face.

## Repository Layout

- `configs/`  
  Pipeline configuration files.
- `docs/`  
  Protocol notes and short release documents.
- `metadata/`  
  Schemas, ship-type tables, split metadata, and auxiliary manifests.
- `scripts/`  
  End-to-end preprocessing stages from raw AIS to benchmark export.
- `benchmark/`  
  Preview shards, export summaries, manifests, and a minimal dataset wrapper.
- `mini_benchmark/`  
  Compact benchmark builders and released compact subsets.
- `NOAA/`  
  NOAA-specific release notes, scripts, and compact metadata.
- `visualization/`  
  Utilities and example figures for inspecting samples and context.
- `words_for_paper/`  
  Paper drafting notes and dataset statistics prepared during the release cycle.

## Released Assets

This repository includes:

- the preprocessing code for the full monthly pipeline
- release-ready scripts for compact subset construction
- lightweight preview shards for schema inspection
- the `ship_core_lite` compact release
- the code, reports, and IDs for `clean_ship_core_lite_v1`
- visualization scripts and sample outputs

The full benchmark payload is intentionally stored outside GitHub.

## Dataset Layout on Hugging Face

The Hugging Face release is organized by source:

- `DMA/benchmark/core`
- `DMA/benchmark/full`
- `DMA/mini_benchmark/ship_core_lite`
- `DMA/mini_benchmark/clean_ship_core_lite_v1`
- `NOAA/benchmark/core`
- `NOAA/benchmark/full`
- `NOAA/mini_bench/clean_ship_core_lite_v1`

For the DMA compact clean subset, the release also includes:

- `environment_v1`
- `environment_v2`
- `social_env_v1`

Raw source files are provided for reproducibility:

- `DMA/data_raw/dma/incoming/2025-09/aisdk-2025-09-*.zip`
- `NOAA/data_raw/noaa/2025-03/ais-2025-03-*.csv.zst`

## Quick Start

Inspect a preview shard:

```bash
python - <<'PY'
from benchmark.ship_trajectory_dataset import ShipTrajectoryDataset
ds = ShipTrajectoryDataset("benchmark", version="core_preview", split="train")
print(len(ds), ds[0]["hist"].shape, ds[0]["future"].shape)
PY
```

Run the monthly DMA pipeline:

```bash
bash scripts/run_pipeline_parallel_dma_2025_09.sh
```

Build the representative compact subset:

```bash
bash mini_benchmark/run_build_ship_core_lite.sh
```

Build the quality-first compact subset:

```bash
bash mini_benchmark/run_build_clean_ship_core_lite.sh
```

Build the environment packages for `clean_ship_core_lite_v1`:

```bash
bash mini_benchmark/run_build_clean_ship_core_lite_environment_v1.sh
bash mini_benchmark/run_build_clean_ship_core_lite_environment_v2.sh
```

## Git and Data Hosting

The following stay in Git:

- `configs/`
- `docs/`
- `metadata/`
- `scripts/`
- `benchmark/`
- `mini_benchmark/`
- `NOAA/`
- `visualization/`
- `words_for_paper/`
- `README.md`
- `LICENSE`
- `CITATION.cff`

The following stay out of Git and live on Hugging Face:

- `hf_release/`
- full benchmark shards
- full raw monthly archives

## License and Citation

The code is released under the MIT License. Use of the benchmark should also follow the terms attached to the original AIS sources and the release notes provided with the dataset.

For citation metadata, see:

- `CITATION.cff`
