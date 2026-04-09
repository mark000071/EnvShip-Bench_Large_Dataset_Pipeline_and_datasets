# clean_ship_core_lite_v1

`clean_ship_core_lite_v1` is the quality-first compact release derived from the DMA `benchmark/core` split.

Unlike `ship_core_lite`, which remains broadly representative, this subset is tuned for cleaner and more controlled experiments. It favors windows with stronger continuity, lower interpolation burden, smoother local dynamics, and more stable short-term motion.

## Release contents

- `build_clean_ship_core_lite.py`
  Builder snapshot used for this release.
- `train/part-000.csv.gz`
- `val/part-000.csv.gz`
- `test/part-000.csv.gz`
- `sample_ids/*.txt`
- `reports/*_selected_metadata.csv`
- `reports/*_report.json`
- `summary.json`

## Target size

- train: 19,500
- val: 2,400
- test: 2,100

## Selection policy

The clean subset applies a stricter policy than the main compact release:

- keep `cargo_tanker` and `passenger_ferry`
- exclude noisier or less stable groups such as `other_unknown`, `fishing`, `tug_service`, and `sailing_leisure`
- reject windows with weak continuity, excessive interpolation, drifting behavior, unstable motion cues, or abrupt bridge turns
- favor smoother, moderate-speed trajectories with more regular short-term behavior
- cap repeated windows with `max_per_mmsi=12`, `max_per_segment=3`, and `min_segment_step_gap=20`

## Rebuilding

```bash
bash mini_benchmark/run_build_clean_ship_core_lite.sh
```
