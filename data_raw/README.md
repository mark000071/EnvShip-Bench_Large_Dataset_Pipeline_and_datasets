# Raw Data

This repository does not store the full DMA raw AIS archive on GitHub.

For the public release, raw data are distributed through the companion Hugging Face dataset page:

- <https://huggingface.co/datasets/mark000071/EnvShip-Bench_An_Environment-Enhanced_Benchmark_for_Short-Term_Vessel_Trajectory_Prediction>

The preprocessing scripts expect the monthly DMA ZIP files under:

```text
data_raw/dma/incoming/2025-09/
```

At minimum, the Hugging Face release includes one reference raw file:

- `DMA/data_raw/dma/incoming/2025-09/aisdk-2025-09-01.zip`

To reproduce the full monthly benchmark, download the DMA monthly archive to the local `data_raw/` tree before running the pipeline scripts.
