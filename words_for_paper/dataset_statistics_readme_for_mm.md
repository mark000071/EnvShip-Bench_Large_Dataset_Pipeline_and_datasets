# Dataset Statistics and Profiling Notes for MM Submission

This document summarizes the most important dataset statistics for the current EnvShip-Bench release, with emphasis on the large-scale DMA-backed `benchmark/core`, the quality-first compact subset `clean_ship_core_lite_v1`, and the context packages built on top of the compact subset, including `environment_v1`, `environment_v2`, `environment_v3`, and `social_env_v1`.

The goal of this document is to provide a paper-ready statistical overview for the dataset section of an ACM Multimedia submission. The wording is intentionally academic and can be adapted into the benchmark overview, dataset statistics, or supplementary material.

---

## 1. Overview

The current release is organized into two principal benchmark layers.

1. A **large-scale standardized trajectory benchmark** built from raw AIS data released by the Danish Maritime Authority (DMA), exported as `benchmark/core` and `benchmark/full`.
2. A **quality-first compact benchmark**, `clean_ship_core_lite_v1`, derived from `benchmark/core` through stricter motion-quality screening, vessel-type restriction, representative stratified sampling, and redundancy control.

On top of `clean_ship_core_lite_v1`, the project further provides:

- `environment_v1`: a first sample-centric environment package with shoreline and waterfront descriptors;
- `environment_v2`: an extended mixed environment package with larger local patches, richer raster semantics, and stronger geometric descriptors;
- `environment_v3`: a localized variant of the environment package optimized for near-anchor cues;
- `social_env_v1`: a synchronized target-centric social context package with nearby-vessel trajectories and relative-motion descriptors.

Unless otherwise noted, all samples follow the same unified forecasting protocol:

- observation horizon: 10 minutes
- prediction horizon: 10 minutes
- temporal interval: 20 seconds
- history length: 30 points
- future length: 30 points

---

## 2. DMA Core Benchmark (`benchmark/core`)

### 2.1 Split Sizes

The large-scale core benchmark contains the following numbers of released samples:

| Split | Samples |
|---|---:|
| Train | 4,429,982 |
| Val | 521,170 |
| Test | 531,124 |

In total, the current `benchmark/core` release contains **5,482,276** core-eligible trajectory windows.

For reference, the corresponding `benchmark/full` release contains:

| Split | Samples |
|---|---:|
| Train | 4,487,077 |
| Val | 528,665 |
| Test | 538,270 |

Thus, the strict core benchmark retains the vast majority of usable windows, while excluding the noisiest or least reliable candidates.

### 2.2 Split-Wise Vessel and Segment Counts

The split-wise diversity of the core benchmark is summarized below:

| Split | #Samples | #Unique MMSI | #Unique Segments |
|---|---:|---:|---:|
| Train | 4,429,982 | 7,558 | 124,043 |
| Val | 521,170 | 910 | 13,649 |
| Test | 531,124 | 870 | 15,374 |

These numbers indicate that the benchmark is not driven by a small number of very long trajectories; instead, it covers a large number of vessels and many independent motion segments, especially in the training split.

### 2.3 Major Vessel-Type Composition

The dominant vessel types in the current `benchmark/core` release are as follows.

**Train**

- cargo: 1,878,092
- tanker: 780,149
- fishing: 632,710
- sailing: 400,638
- passenger: 199,958
- pleasure: 193,884
- other: 81,917
- military: 67,144
- tug: 60,291

**Val**

- cargo: 227,848
- tanker: 88,610
- fishing: 69,278
- sailing: 39,555
- passenger: 24,401
- pleasure: 20,131
- other: 18,544
- tug: 8,428
- military: 8,251
- dredging: 5,178

**Test**

- cargo: 212,652
- fishing: 77,601
- tanker: 75,271
- sailing: 52,297
- passenger: 34,084
- pleasure: 17,670
- other: 14,497
- tug: 12,337
- law enforcement: 10,011

Overall, the large-scale core benchmark is strongly dominated by merchant traffic, especially cargo and tanker vessels, but it still retains a wide range of additional maritime behaviors such as fishing, passenger, sailing, tug, and other specialized categories. This makes the full core benchmark suitable for general-purpose vessel trajectory prediction rather than only merchant-route forecasting.

### 2.4 Interpolation Burden

Across the full quality-labeled pool before final split export, the weighted mean true interpolation burden is:

- **mean true interpolation ratio**: **0.00781**

This value is computed from the stage-12 quality-labeled windows and indicates that, on average, less than 1% of the released trajectory points are true short-gap imputations. In practice, this confirms that the benchmark is not primarily constructed from aggressively reconstructed trajectories. Instead, most windows are formed from directly observed motion with only a small amount of short-gap completion.

At the full quality-label stage, the total sample accounting is:

- total windows processed: 5,574,929
- core-eligible: 5,482,276
- full-only: 71,736
- dropped: 20,917

This confirms that the final DMA core release is both large and clean, with relatively low interpolation burden and only modest rejection at the final sample-quality stage.

---

## 3. Compact Benchmark (`clean_ship_core_lite_v1`)

### 3.1 Motivation and Role

`clean_ship_core_lite_v1` is a compact, quality-first derivative benchmark constructed from `benchmark/core`. It is not intended to mirror the full distribution of the large-scale benchmark. Instead, it is designed for:

- efficient experimentation,
- controlled benchmarking,
- low-noise trajectory modeling,
- environment-aware modeling,
- interaction-aware augmentation on top of a stable and reproducible target set.

This subset uses stricter motion-quality filters and retains only two major vessel groups:

- `cargo_tanker`
- `passenger_ferry`

The following groups are excluded:

- `fishing`
- `other_unknown`
- `sailing_leisure`
- `tug_service`

### 3.2 Split Sizes

The compact benchmark is released with the following fixed sizes:

| Split | Samples |
|---|---:|
| Train | 19,500 |
| Val | 2,400 |
| Test | 2,100 |

Total compact benchmark size:

- **24,000 samples**

### 3.3 Vessel and Segment Counts

| Split | #Samples | #Unique MMSI | #Unique Segments |
|---|---:|---:|---:|
| Train | 19,500 | 1,953 | 9,379 |
| Val | 2,400 | 237 | 1,106 |
| Test | 2,100 | 204 | 1,025 |

These numbers show that even the compact benchmark retains substantial vessel and segment diversity, especially given its quality-first design and explicit redundancy control.

### 3.4 Vessel-Type Composition

The compact benchmark is intentionally concentrated on clean merchant and ferry trajectories.

**Train**

- cargo_tanker: 18,744
- passenger_ferry: 756

**Val**

- cargo_tanker: 2,335
- passenger_ferry: 65

**Test**

- cargo_tanker: 2,020
- passenger_ferry: 80

This composition is expected and reflects the design goal of prioritizing stable and interpretable vessel motion.

### 3.5 Speed and Motion Composition

The speed distribution is strongly centered on moderate commercial operating regimes.

**Train speed bins**

- 10–15 kn: 11,818
- 5–10 kn: 6,499
- 15–20 kn: 1,055
- 0–5 kn: 119
- >20 kn: 9

**Val speed bins**

- 10–15 kn: 1,517
- 5–10 kn: 789
- 15–20 kn: 80
- 0–5 kn: 13
- >20 kn: 1

**Test speed bins**

- 10–15 kn: 1,298
- 5–10 kn: 659
- 15–20 kn: 136
- 0–5 kn: 7

The motion-scale composition is also highly concentrated on strong forward motion:

**Train**

- high-motion windows: 19,326
- mid-motion windows: 174

**Val**

- high-motion windows: 2,383
- mid-motion windows: 17

**Test**

- high-motion windows: 2,086
- mid-motion windows: 14

This concentration is a direct consequence of the quality-first construction strategy. The compact benchmark is intentionally biased toward stable, moderately fast, forward-moving vessel behavior rather than low-motion or ambiguous maneuvering cases.

### 3.6 Quality Profile

The compact benchmark exhibits a very strong quality profile across all splits.

**Train**

- mean quality score: 95.6679
- mean difficulty score: 3.9028
- mean average speed: 10.9643 kn
- mean historical speed CV: 0.0322
- mean historical efficiency: 0.9982
- mean future efficiency: 0.9996
- mean future linearity: 0.9665
- mean bridge-turn angle: 1.6849 deg

**Val**

- mean quality score: 95.7438
- mean difficulty score: 3.9281
- mean average speed: 10.8514 kn
- mean historical speed CV: 0.0313
- mean historical efficiency: 0.9982
- mean future efficiency: 0.9996
- mean future linearity: 0.9665

**Test**

- mean quality score: 95.7228
- mean difficulty score: 3.9181
- mean average speed: 11.1113 kn
- mean historical speed CV: 0.0312
- mean historical efficiency: 0.9982
- mean future efficiency: 0.9996
- mean future linearity: 0.9665
- mean bridge-turn angle: 1.7413 deg

These numbers confirm that `clean_ship_core_lite_v1` is not merely smaller than the full benchmark, but substantially cleaner and more controlled.

### 3.7 Interpolation Burden

The compact benchmark has an extremely small interpolation burden.

| Split | Mean `interp_ratio_total` | Median | P90 | Max |
|---|---:|---:|---:|---:|
| Train | 0.00092 | 0.0 | 0.0 | 0.10 |
| Val | 0.00087 | 0.0 | 0.0 | 0.0833 |
| Test | 0.00135 | 0.0 | 0.0 | 0.10 |

In all three splits, the median and 90th percentile are exactly zero, meaning that the vast majority of compact samples contain no true interpolated points at all. This is one of the key reasons why `clean_ship_core_lite_v1` is particularly suitable for controlled trajectory-prediction experiments.

### 3.8 Selection Pressure

The compact subset is highly selective relative to the full core benchmark:

- train candidates: 398,535 eligible from 4,429,982 input rows
- val candidates: 48,954 eligible from 521,170 input rows
- test candidates: 48,801 eligible from 531,124 input rows

The largest rejection categories are:

- excluded ship groups,
- poor consistency between COG and actual motion direction,
- excessive bridge turning,
- insufficient future linearity,
- insufficient future efficiency,
- excessive interpolation burden,
- out-of-range speed profiles.

This makes the compact benchmark an explicitly curated release rather than a naive downsampled subset.

---

## 4. Environment Context Packages

### 4.1 `environment_v1`

`environment_v1` is the first environment extension for `clean_ship_core_lite_v1`. It uses a smaller local patch and provides sample-centric descriptors related to shoreline and waterfront proximity, together with vector and raster context.

Coverage is complete with respect to the clean compact benchmark:

| Split | #Samples | Augmented Rows |
|---|---:|---:|
| Train | 19,500 | 19,500 |
| Val | 2,400 | 2,400 |
| Test | 2,100 | 2,100 |

Shoreline and waterfront coverage statistics are:

| Split | Shoreline Positive | Waterfront Positive |
|---|---:|---:|
| Train | 718 | 515 |
| Val | 88 | 52 |
| Test | 83 | 71 |

Average nearest distances:

- Train: mean minimum shoreline distance = 3977.55 m; mean minimum waterfront distance = 4016.22 m
- Val: mean minimum shoreline distance = 3984.55 m; mean minimum waterfront distance = 4029.57 m
- Test: mean minimum shoreline distance = 3955.45 m; mean minimum waterfront distance = 3989.97 m

The scene-group summary of `environment_v1` shows that the compact benchmark remains heavily dominated by offshore-route samples:

- offshore_route: 23,046
- harbor_port: 437
- nearshore_channel: 353
- coastal_route: 164

This is an important observation for paper discussion: the clean compact subset is intentionally high quality, but it is not geographically balanced across all maritime scene types.

### 4.2 `environment_v2`

`environment_v2` is a stronger mixed environment package built on the same 24,000 compact samples. It expands the local patch radius and enriches the geometric representation.

Core setup:

- patch radius: 5,000 m
- grid size: 128 × 128
- tile size: 0.25 degrees

Released raster semantics:

1. land mask
2. water mask
3. geo-navigable mask
4. natural boundary mask
5. manmade boundary mask
6. barrier mask
7. signed distance to shoreline/barrier
8. signed distance to navigable region

Descriptor columns include:

- nearest shoreline distance
- nearest manmade-boundary distance
- nearest barrier distance
- water ratio
- navigable ratio
- boundary density
- natural boundary density
- manmade boundary density
- barrier density
- anchor-in-water and anchor-in-navigable flags
- scene type
- environment quality score

Coverage statistics for `environment_v2` over all 24,000 compact samples:

- total rows: 24,000
- anchor in water: 23,940
- anchor in navigable region: 23,940
- samples with natural boundary evidence: 2,818
- samples with manmade boundary evidence: 1,881

Scene-type distribution:

- open_water: 23,408
- nearshore: 367
- constrained: 190
- harbor: 35

Environment descriptor profile:

- mean environment quality score: 0.9916
- median environment quality score: 1.0
- mean nearest shoreline distance: 4828.26 m
- mean nearest manmade-boundary distance: 4858.36 m

These statistics indicate that `environment_v2` provides complete and high-confidence context for the compact benchmark, but the resulting sample population remains predominantly open-water oriented. This should be explicitly acknowledged in the paper rather than hidden.

### 4.3 `environment_v3`

`environment_v3` is an incremental environment package derived from `environment_v2`, intended to provide more local, near-anchor cues.

Core setup:

- inherited from `environment_v2`
- local patch radius: 2,000 m
- grid size: 128 × 128

Coverage is again complete over the compact benchmark:

| Split | #Samples |
|---|---:|
| Train | 19,500 |
| Val | 2,400 |
| Test | 2,100 |

`environment_v3` should be described as a localized refinement of the environment representation rather than a different sample release. It preserves the same split membership and target trajectories while focusing more tightly on near-anchor navigability structure.

---

## 5. Social Context Package (`social_env_v1`)

`social_env_v1` augments the clean compact benchmark with synchronized target-centric nearby-vessel context.

Construction setup:

- neighborhood radius: 3,000 m
- maximum exported neighbors: 8
- interaction-candidate threshold: at least 2 neighbors
- snapshot bucket count: 64

### 5.1 Coverage

Coverage is complete with respect to the compact benchmark:

| Split | #Samples | Social Context Available |
|---|---:|---:|
| Train | 19,500 | 19,500 |
| Val | 2,400 | 2,400 |
| Test | 2,100 | 2,100 |

This means every compact sample is assigned a synchronized social-context record, even when no neighboring vessels are present.

### 5.2 Neighbor Coverage and Interaction Candidates

Split-wise statistics:

| Split | Samples with Neighbors | Interaction Candidates | Mean Neighbor Count |
|---|---:|---:|---:|
| Train | 3,506 | 529 | 0.2103 |
| Val | 388 | 40 | 0.1804 |
| Test | 338 | 42 | 0.1838 |

Across the full 24,000-sample compact benchmark:

- samples with at least one neighbor: 4,232
- interaction candidates: 611

Density-bin distribution over all compact samples:

- isolated: 19,768
- sparse: 4,224
- medium: 8

Thus, `social_env_v1` should be described as a **target-centric social context layer** rather than a fully dense multi-agent scene benchmark. The majority of the compact benchmark remains socially sparse, which is consistent with its merchant- and ferry-focused, quality-first construction.

### 5.3 Relative Distance and Risk Summary

Across all 24,000 samples in `social_env_v1`:

- mean minimum neighbor distance: 1915.09 m
- median minimum neighbor distance: 1954.84 m
- 90th percentile minimum neighbor distance: 2782.28 m
- minimum observed neighbor distance: 33.91 m

CPA summary:

- mean minimum CPA: 1258.45 m
- median minimum CPA: 1200.95 m
- 90th percentile minimum CPA: 2204.60 m
- minimum observed CPA: 0.046 m

These numbers indicate that the social package captures a wide range of interaction strengths, from clearly separated co-presence to a small number of very close vessel encounters.

---

## 6. Recommended Positioning in the Paper

The statistics above suggest the following paper-safe interpretation.

### 6.1 For the DMA Core Benchmark

The large-scale core release should be positioned as:

- a standardized short-term vessel trajectory benchmark,
- large in sample count,
- broad in vessel-type coverage,
- low in interpolation burden,
- appropriate for general vessel trajectory prediction.

### 6.2 For `clean_ship_core_lite_v1`

The compact benchmark should be positioned as:

- a quality-first subset,
- focused on merchant and ferry traffic,
- extremely low-noise,
- extremely low-interpolation,
- suitable for reproducible and efficient experimentation,
- especially suitable for context-aware extensions.

### 6.3 For `environment_v1/v2/v3`

The environment packages should be positioned as:

- complete context augmentations of the compact benchmark,
- progressively richer environment representations,
- with `environment_v2` as the strongest mixed representation,
- and `environment_v3` as the more localized refinement.

### 6.4 For `social_env_v1`

The social package should be positioned as:

- a synchronized target-centric social annotation layer,
- suitable for interaction-aware analysis,
- but not yet a dense scene-centric social benchmark in the pedestrian-dataset sense.

This distinction is important for accurate comparison with datasets such as SDD or ETH/UCY.

---

## 7. One-Paragraph Summary

The current EnvShip-Bench release consists of a large DMA-backed core trajectory benchmark with 5.48M core-eligible windows and a quality-first compact derivative benchmark, `clean_ship_core_lite_v1`, with 24K carefully curated merchant and ferry samples. The large-scale core release is broad in vessel-type coverage and maintains a low true interpolation burden, while the compact benchmark exhibits an especially clean motion profile, near-zero interpolation for the vast majority of samples, and strong diversity in vessels and trajectory segments despite its reduced scale. On top of the compact benchmark, `environment_v1`, `environment_v2`, and `environment_v3` provide complete environment-aware augmentations with shoreline, manmade-boundary, and navigability descriptors, while `social_env_v1` provides synchronized nearby-vessel context and interaction flags. Together, these components support a progression from large-scale trajectory-only forecasting to controlled environment-aware and interaction-aware maritime prediction.
