# Benchmark Assets

This directory contains the lightweight benchmark assets that accompany the public code release.

## Contents

- `export_summary.json`
  Split-level statistics for the full monthly export used in the release build.
- `core_manifest.csv`
  File manifest for the full `core` release.
- `full_manifest.csv`
  File manifest for the full `full` release.
- `core_preview/`
  Small preview shards sampled from the full `core` benchmark.
- `full_preview/`
  Small preview shards sampled from the full `full` benchmark.
- `ship_trajectory_dataset.py`
  A minimal PyTorch dataset wrapper for exported benchmark shards.

## Why only previews are stored here

The complete monthly benchmark is hosted on Hugging Face. The full shard files are too large for a practical GitHub release, so this repository keeps only reduced previews that preserve the released schema and sample layout.

The full benchmark follows the standard split layout:

- `core/train`, `core/val`, `core/test`
- `full/train`, `full/val`, `full/test`

## Intended use

The preview files are useful for:

- checking the exported field schema
- testing data loaders
- inspecting sample structure
- lightweight collaboration without downloading the full release

For model training or large-scale evaluation, use the Hugging Face release or regenerate the benchmark from the preprocessing pipeline.
