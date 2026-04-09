#!/usr/bin/env python3
"""Add sample-centric environment context to clean_ship_core_lite_v1.

This builder keeps the original clean benchmark untouched and writes a paired
environment package under the dataset root:

- anchors: recovered anchor lat/lon at hist_end_ts for every sample
- features: numeric shoreline / waterfront context features
- vectors: clipped local polylines in target-centered meters
- rasters: sparse 2-channel masks (shoreline, manmade waterfront)
- augmented_splits: original sample rows plus numeric environment features

The implementation is intentionally lightweight:
- no geopandas / shapely dependency
- OSM geometry fetched from Overpass and cached per coarse geo tile
- geometry operations are done with simple numpy / math utilities
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import requests


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.pipeline_utils import latlon_to_local_xy


DEFAULT_CLEAN_ROOT = Path(
    "/mnt/nfs/kun/DeepJSCC/NOAA_ship_trajectory_datasets/mini_bench/clean_ship_core_lite_v1"
)
DEFAULT_STAGE10_DIR = Path(
    "/mnt/nfs/kun/DeepJSCC/NOAA_ship_trajectory_datasets/data_interim/10_minlen_filtered/partitions"
)

OVERPASS_URL = ",".join(
    [
        "https://overpass.private.coffee/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://lz4.overpass-api.de/api/interpreter",
        "https://z.overpass-api.de/api/interpreter",
    ]
)
EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class FeatureWay:
    osm_id: int
    category: str
    subtype: str
    lat: tuple[float, ...]
    lon: tuple[float, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-root", type=Path, default=DEFAULT_CLEAN_ROOT)
    parser.add_argument("--stage10-dir", type=Path, default=DEFAULT_STAGE10_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overpass-url", type=str, default=OVERPASS_URL)
    parser.add_argument("--tile-deg", type=float, default=0.25)
    parser.add_argument("--patch-radius-m", type=float, default=2048.0)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--line-sample-step-m", type=float, default=24.0)
    parser.add_argument("--chunk-size", type=int, default=200000)
    parser.add_argument("--max-samples-per-split", type=int, default=0)
    parser.add_argument("--max-tiles", type=int, default=0)
    parser.add_argument("--sleep-between-queries", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=5000)
    return parser.parse_args()


def parse_overpass_urls(value: str) -> list[str]:
    urls = [item.strip() for item in str(value).split(",") if item.strip()]
    if not urls:
        raise ValueError("At least one Overpass endpoint must be provided")
    return urls


def patch_margin_deg_lat(meters: float) -> float:
    return math.degrees(meters / EARTH_RADIUS_M)


def patch_margin_deg_lon(meters: float, ref_lat: float) -> float:
    denom = EARTH_RADIUS_M * max(math.cos(math.radians(ref_lat)), 1e-6)
    return math.degrees(meters / denom)


def safe_json_dumps(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def clip_value(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def point_to_segment_distance(
    px: float, py: float, x0: float, y0: float, x1: float, y1: float
) -> float:
    vx = x1 - x0
    vy = y1 - y0
    norm2 = vx * vx + vy * vy
    if norm2 <= 1e-12:
        return math.hypot(px - x0, py - y0)
    t = ((px - x0) * vx + (py - y0) * vy) / norm2
    t = clip_value(t, 0.0, 1.0)
    proj_x = x0 + t * vx
    proj_y = y0 + t * vy
    return math.hypot(px - proj_x, py - proj_y)


def segment_length(x0: float, y0: float, x1: float, y1: float) -> float:
    return math.hypot(x1 - x0, y1 - y0)


def local_bbox_intersects(points_x: np.ndarray, points_y: np.ndarray, radius_m: float) -> bool:
    return not (
        np.max(points_x) < -radius_m
        or np.min(points_x) > radius_m
        or np.max(points_y) < -radius_m
        or np.min(points_y) > radius_m
    )


def _compute_outcode(x: float, y: float, radius: float) -> int:
    code = 0
    if x < -radius:
        code |= 1
    elif x > radius:
        code |= 2
    if y < -radius:
        code |= 4
    elif y > radius:
        code |= 8
    return code


def clip_segment_to_square(
    x0: float, y0: float, x1: float, y1: float, radius: float
) -> tuple[float, float, float, float] | None:
    out0 = _compute_outcode(x0, y0, radius)
    out1 = _compute_outcode(x1, y1, radius)
    while True:
        if not (out0 | out1):
            return x0, y0, x1, y1
        if out0 & out1:
            return None
        out = out0 or out1
        if out & 8:
            x = x0 + (x1 - x0) * (radius - y0) / (y1 - y0)
            y = radius
        elif out & 4:
            x = x0 + (x1 - x0) * (-radius - y0) / (y1 - y0)
            y = -radius
        elif out & 2:
            y = y0 + (y1 - y0) * (radius - x0) / (x1 - x0)
            x = radius
        else:
            y = y0 + (y1 - y0) * (-radius - x0) / (x1 - x0)
            x = -radius
        if out == out0:
            x0, y0 = x, y
            out0 = _compute_outcode(x0, y0, radius)
        else:
            x1, y1 = x, y
            out1 = _compute_outcode(x1, y1, radius)


def rasterize_segment(
    mask: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    radius_m: float,
    sample_step_m: float,
) -> None:
    length = segment_length(x0, y0, x1, y1)
    num = max(2, int(math.ceil(length / max(sample_step_m, 1.0))) + 1)
    xs = np.linspace(x0, x1, num=num)
    ys = np.linspace(y0, y1, num=num)
    grid_size = mask.shape[0]
    gx = np.floor((xs + radius_m) / (2.0 * radius_m) * grid_size).astype(int)
    gy = np.floor((ys + radius_m) / (2.0 * radius_m) * grid_size).astype(int)
    valid = (gx >= 0) & (gx < grid_size) & (gy >= 0) & (gy < grid_size)
    mask[gy[valid], gx[valid]] = 1


def tile_id_for_point(lat: float, lon: float, tile_deg: float) -> str:
    tile_lat = math.floor(lat / tile_deg) * tile_deg
    tile_lon = math.floor(lon / tile_deg) * tile_deg
    return f"{tile_lat:+08.3f}_{tile_lon:+09.3f}"


def tile_origin_from_id(tile_id: str) -> tuple[float, float]:
    tile_lat, tile_lon = tile_id.split("_")
    return float(tile_lat), float(tile_lon)


def make_overpass_query(
    south: float, west: float, north: float, east: float, timeout_seconds: int = 45
) -> str:
    # Ways only keeps parsing simple and is sufficient for the first release.
    return f"""
[out:json][timeout:{timeout_seconds}];
(
  way["natural"="coastline"]({south},{west},{north},{east});
  way["waterway"~"riverbank|dock|canal"]({south},{west},{north},{east});
  way["man_made"~"pier|breakwater|groyne|quay"]({south},{west},{north},{east});
  way["landuse"="port"]({south},{west},{north},{east});
);
out geom;
""".strip()


def classify_way(tags: dict[str, str]) -> tuple[str, str] | None:
    natural = tags.get("natural", "")
    waterway = tags.get("waterway", "")
    man_made = tags.get("man_made", "")
    landuse = tags.get("landuse", "")
    if natural == "coastline":
        return "shoreline", "coastline"
    if waterway in {"riverbank", "dock", "canal"}:
        return "shoreline", waterway
    if man_made in {"pier", "breakwater", "groyne", "quay"}:
        return "waterfront", man_made
    if landuse == "port":
        return "waterfront", "port_boundary"
    return None


def parse_osm_ways(payload: dict) -> list[FeatureWay]:
    ways: list[FeatureWay] = []
    for element in payload.get("elements", []):
        if element.get("type") != "way":
            continue
        classification = classify_way(element.get("tags", {}))
        if classification is None:
            continue
        geom = element.get("geometry", [])
        if len(geom) < 2:
            continue
        lat = tuple(float(node["lat"]) for node in geom)
        lon = tuple(float(node["lon"]) for node in geom)
        category, subtype = classification
        ways.append(
            FeatureWay(
                osm_id=int(element["id"]),
                category=category,
                subtype=subtype,
                lat=lat,
                lon=lon,
            )
        )
    return ways


def fetch_tile_osm(
    tile_id: str,
    tile_points: pd.DataFrame,
    cache_dir: Path,
    overpass_url: str,
    patch_radius_m: float,
    sleep_seconds: float,
) -> list[FeatureWay]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{tile_id}.json"
    if cache_path.exists():
        return parse_osm_ways(json.loads(cache_path.read_text(encoding="utf-8")))

    south = float(tile_points["anchor_lat"].min())
    north = float(tile_points["anchor_lat"].max())
    west = float(tile_points["anchor_lon"].min())
    east = float(tile_points["anchor_lon"].max())
    center_lat = 0.5 * (south + north)
    lat_margin = patch_margin_deg_lat(patch_radius_m * 1.4)
    lon_margin = patch_margin_deg_lon(patch_radius_m * 1.4, center_lat)
    query = make_overpass_query(
        south=south - lat_margin,
        west=west - lon_margin,
        north=north + lat_margin,
        east=east + lon_margin,
        timeout_seconds=45,
    )

    overpass_urls = parse_overpass_urls(overpass_url)
    last_error: Exception | None = None
    for attempt in range(4):
        for endpoint_idx, endpoint in enumerate(overpass_urls, start=1):
            try:
                response = requests.post(
                    endpoint,
                    data=query.encode("utf-8"),
                    headers={"Content-Type": "text/plain; charset=utf-8"},
                    timeout=60,
                )
                response.raise_for_status()
                payload = response.json()
                cache_path.write_text(json.dumps(payload), encoding="utf-8")
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
                return parse_osm_ways(payload)
            except Exception as exc:
                last_error = exc
                print(
                    f"[osm] tile={tile_id} attempt={attempt + 1} endpoint={endpoint_idx}/{len(overpass_urls)} failed: {exc}"
                )
        wait_seconds = min(30.0, 2.0 * (attempt + 1))
        print(f"[osm] tile={tile_id} retrying in {wait_seconds:.1f}s")
        time.sleep(wait_seconds)
    print(f"[osm] tile={tile_id} failed after retries; using empty OSM payload. last_error={last_error}")
    empty_payload = {"elements": []}
    cache_path.write_text(json.dumps(empty_payload), encoding="utf-8")
    return []


def load_clean_split_rows(split_path: Path, max_rows: int = 0) -> pd.DataFrame:
    rows = pd.read_csv(
        split_path,
        compression="gzip",
        usecols=["sample_id", "mmsi", "segment_id", "hist_end_ts", "pred_end_ts"],
        low_memory=False,
    )
    if max_rows > 0:
        rows = rows.iloc[:max_rows].copy()
    rows["hist_end_ts"] = pd.to_datetime(rows["hist_end_ts"], utc=True)
    rows["pred_end_ts"] = pd.to_datetime(rows["pred_end_ts"], utc=True)
    rows["split"] = split_path.parent.name
    return rows


def collect_clean_samples(clean_root: Path, max_samples_per_split: int) -> pd.DataFrame:
    frames = []
    for split in ("train", "val", "test"):
        path = clean_root / split / "part-000.csv.gz"
        frame = load_clean_split_rows(path, max_rows=max_samples_per_split)
        frames.append(frame)
    samples = pd.concat(frames, ignore_index=True)
    return samples


def recover_anchor_points(
    samples: pd.DataFrame,
    stage10_dir: Path,
    chunk_size: int,
    tile_deg: float,
) -> pd.DataFrame:
    samples = samples.copy()
    samples["hist_end_ns"] = samples["hist_end_ts"].astype("int64")
    partition_files = sorted(stage10_dir.glob("part-*.csv.gz"))
    requests_by_partition: dict[int, dict[tuple[str, int], list[int]]] = defaultdict(lambda: defaultdict(list))
    needed_segments_by_partition: dict[int, set[str]] = defaultdict(set)
    for idx, row in samples.iterrows():
        part_id = int(row["partition_id"])
        key = (str(row["segment_id"]), int(row["hist_end_ns"]))
        requests_by_partition[part_id][key].append(idx)
        needed_segments_by_partition[part_id].add(str(row["segment_id"]))

    anchors = np.full((len(samples), 2), np.nan, dtype=np.float64)
    for part_id, path in enumerate(partition_files):
        request_map = requests_by_partition.get(part_id)
        if not request_map:
            continue
        segment_ids = needed_segments_by_partition[part_id]
        print(f"[anchors] part={part_id:03d} scanning {path.name} requested_segments={len(segment_ids)}")
        for chunk in pd.read_csv(
            path,
            compression="gzip",
            usecols=["segment_id", "timestamp_utc", "lat", "lon"],
            chunksize=chunk_size,
            low_memory=False,
        ):
            chunk = chunk.loc[chunk["segment_id"].isin(segment_ids)].copy()
            if chunk.empty:
                continue
            chunk["timestamp_utc"] = pd.to_datetime(chunk["timestamp_utc"], utc=True)
            chunk["ts_ns"] = chunk["timestamp_utc"].astype("int64")
            for row in chunk.itertuples(index=False):
                key = (str(row.segment_id), int(row.ts_ns))
                target_indices = request_map.get(key)
                if not target_indices:
                    continue
                for sample_idx in target_indices:
                    anchors[sample_idx, 0] = float(row.lat)
                    anchors[sample_idx, 1] = float(row.lon)

    samples["anchor_lat"] = anchors[:, 0]
    samples["anchor_lon"] = anchors[:, 1]
    missing = samples["anchor_lat"].isna().sum()
    if missing:
        missing_samples = samples.loc[samples["anchor_lat"].isna(), "sample_id"].head(10).tolist()
        raise RuntimeError(f"Failed to recover anchors for {missing} samples; examples: {missing_samples}")
    samples["tile_id"] = [
        tile_id_for_point(lat, lon, tile_deg=tile_deg)
        for lat, lon in zip(samples["anchor_lat"], samples["anchor_lon"], strict=False)
    ]
    return samples


def clip_way_to_local_segments(
    way: FeatureWay,
    anchor_lat: float,
    anchor_lon: float,
    radius_m: float,
) -> list[np.ndarray]:
    lat = np.asarray(way.lat, dtype=np.float64)
    lon = np.asarray(way.lon, dtype=np.float64)
    x, y = latlon_to_local_xy(lat, lon, anchor_lat, anchor_lon)
    if not local_bbox_intersects(x, y, radius_m):
        return []

    segments: list[np.ndarray] = []
    for idx in range(1, len(x)):
        clipped = clip_segment_to_square(float(x[idx - 1]), float(y[idx - 1]), float(x[idx]), float(y[idx]), radius_m)
        if clipped is None:
            continue
        x0, y0, x1, y1 = clipped
        segments.append(np.asarray([[x0, y0], [x1, y1]], dtype=np.float32))
    return segments


def summarize_category(
    segments: list[np.ndarray],
    radius_m: float,
) -> dict[str, float]:
    if not segments:
        return {
            "min_dist_m": radius_m * 2.0,
            "segment_count": 0.0,
            "total_length_m": 0.0,
            "near_250m": 0.0,
            "near_500m": 0.0,
            "near_1000m": 0.0,
        }
    min_dist = float("inf")
    total_length = 0.0
    near_250 = 0
    near_500 = 0
    near_1000 = 0
    for segment in segments:
        x0, y0 = map(float, segment[0])
        x1, y1 = map(float, segment[1])
        dist = point_to_segment_distance(0.0, 0.0, x0, y0, x1, y1)
        min_dist = min(min_dist, dist)
        total_length += segment_length(x0, y0, x1, y1)
        if dist <= 250.0:
            near_250 += 1
        if dist <= 500.0:
            near_500 += 1
        if dist <= 1000.0:
            near_1000 += 1
    return {
        "min_dist_m": min_dist if math.isfinite(min_dist) else radius_m * 2.0,
        "segment_count": float(len(segments)),
        "total_length_m": total_length,
        "near_250m": float(near_250),
        "near_500m": float(near_500),
        "near_1000m": float(near_1000),
    }


def build_environment_for_split(
    split_samples: pd.DataFrame,
    tile_way_cache: dict[str, list[FeatureWay]],
    feature_dir: Path,
    vector_dir: Path,
    raster_dir: Path,
    patch_radius_m: float,
    grid_size: int,
    line_sample_step_m: float,
    log_every: int,
) -> dict[str, object]:
    feature_dir.mkdir(parents=True, exist_ok=True)
    vector_dir.mkdir(parents=True, exist_ok=True)
    raster_dir.mkdir(parents=True, exist_ok=True)
    vector_path = vector_dir / "vectors.jsonl.gz"
    feature_rows = []
    raster = np.zeros((len(split_samples), 2, grid_size, grid_size), dtype=np.uint8)
    sample_ids = split_samples["sample_id"].astype(str).to_numpy()

    with gzip.open(vector_path, "wt", encoding="utf-8") as vector_handle:
        for idx, row in enumerate(split_samples.itertuples(index=False), start=0):
            ways = tile_way_cache[str(row.tile_id)]
            shoreline_segments: list[np.ndarray] = []
            waterfront_segments: list[np.ndarray] = []
            vector_payload = {
                "sample_id": str(row.sample_id),
                "split": str(row.split),
                "anchor_lat": float(row.anchor_lat),
                "anchor_lon": float(row.anchor_lon),
                "tile_id": str(row.tile_id),
                "shoreline": [],
                "waterfront": [],
            }

            for way in ways:
                clipped_segments = clip_way_to_local_segments(
                    way=way,
                    anchor_lat=float(row.anchor_lat),
                    anchor_lon=float(row.anchor_lon),
                    radius_m=patch_radius_m,
                )
                if not clipped_segments:
                    continue
                bucket = shoreline_segments if way.category == "shoreline" else waterfront_segments
                bucket.extend(clipped_segments)
                target_key = "shoreline" if way.category == "shoreline" else "waterfront"
                for segment in clipped_segments:
                    rounded = [[round(float(pt[0]), 2), round(float(pt[1]), 2)] for pt in segment.tolist()]
                    vector_payload[target_key].append(
                        {
                            "osm_id": int(way.osm_id),
                            "subtype": str(way.subtype),
                            "xy": rounded,
                        }
                    )

            for segment in shoreline_segments:
                rasterize_segment(
                    mask=raster[idx, 0],
                    x0=float(segment[0, 0]),
                    y0=float(segment[0, 1]),
                    x1=float(segment[1, 0]),
                    y1=float(segment[1, 1]),
                    radius_m=patch_radius_m,
                    sample_step_m=line_sample_step_m,
                )
            for segment in waterfront_segments:
                rasterize_segment(
                    mask=raster[idx, 1],
                    x0=float(segment[0, 0]),
                    y0=float(segment[0, 1]),
                    x1=float(segment[1, 0]),
                    y1=float(segment[1, 1]),
                    radius_m=patch_radius_m,
                    sample_step_m=line_sample_step_m,
                )

            shoreline_stats = summarize_category(shoreline_segments, patch_radius_m)
            waterfront_stats = summarize_category(waterfront_segments, patch_radius_m)
            feature_rows.append(
                {
                    "sample_id": str(row.sample_id),
                    "split": str(row.split),
                    "segment_id": str(row.segment_id),
                    "hist_end_ts": pd.Timestamp(row.hist_end_ts).isoformat(),
                    "anchor_lat": float(row.anchor_lat),
                    "anchor_lon": float(row.anchor_lon),
                    "tile_id": str(row.tile_id),
                    "min_shoreline_dist_m": round(shoreline_stats["min_dist_m"], 3),
                    "shoreline_segment_count": int(shoreline_stats["segment_count"]),
                    "shoreline_total_length_m": round(shoreline_stats["total_length_m"], 3),
                    "shoreline_segments_within_250m": int(shoreline_stats["near_250m"]),
                    "shoreline_segments_within_500m": int(shoreline_stats["near_500m"]),
                    "shoreline_segments_within_1000m": int(shoreline_stats["near_1000m"]),
                    "min_waterfront_dist_m": round(waterfront_stats["min_dist_m"], 3),
                    "waterfront_segment_count": int(waterfront_stats["segment_count"]),
                    "waterfront_total_length_m": round(waterfront_stats["total_length_m"], 3),
                    "waterfront_segments_within_250m": int(waterfront_stats["near_250m"]),
                    "waterfront_segments_within_500m": int(waterfront_stats["near_500m"]),
                    "waterfront_segments_within_1000m": int(waterfront_stats["near_1000m"]),
                    "has_shoreline_in_patch": int(bool(shoreline_segments)),
                    "has_waterfront_in_patch": int(bool(waterfront_segments)),
                    "shoreline_raster_occupancy": int(raster[idx, 0].sum()),
                    "waterfront_raster_occupancy": int(raster[idx, 1].sum()),
                }
            )
            vector_handle.write(safe_json_dumps(vector_payload) + "\n")

            if (idx + 1) % log_every == 0:
                print(f"[env] split={row.split} processed={idx + 1}/{len(split_samples)}")

    features_df = pd.DataFrame(feature_rows)
    features_df.to_csv(feature_dir / "environment_features.csv", index=False)
    np.savez_compressed(
        raster_dir / "environment_rasters.npz",
        sample_ids=sample_ids,
        raster=raster,
    )
    return {
        "num_samples": int(len(split_samples)),
        "shoreline_positive": int((features_df["has_shoreline_in_patch"] > 0).sum()),
        "waterfront_positive": int((features_df["has_waterfront_in_patch"] > 0).sum()),
        "mean_min_shoreline_dist_m": float(features_df["min_shoreline_dist_m"].mean()),
        "mean_min_waterfront_dist_m": float(features_df["min_waterfront_dist_m"].mean()),
    }


def export_anchor_csv(samples: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    samples[
        ["sample_id", "split", "mmsi", "segment_id", "hist_end_ts", "pred_end_ts", "anchor_lat", "anchor_lon", "tile_id"]
    ].to_csv(out_path, index=False)


def export_augmented_split(clean_root: Path, split: str, env_features: pd.DataFrame, output_path: Path) -> int:
    source_path = clean_root / split / "part-000.csv.gz"
    rows = pd.read_csv(source_path, compression="gzip", low_memory=False)
    env_only = env_features[
        [
            "sample_id",
            "anchor_lat",
            "anchor_lon",
            "tile_id",
            "min_shoreline_dist_m",
            "shoreline_segment_count",
            "shoreline_total_length_m",
            "shoreline_segments_within_250m",
            "shoreline_segments_within_500m",
            "shoreline_segments_within_1000m",
            "min_waterfront_dist_m",
            "waterfront_segment_count",
            "waterfront_total_length_m",
            "waterfront_segments_within_250m",
            "waterfront_segments_within_500m",
            "waterfront_segments_within_1000m",
            "has_shoreline_in_patch",
            "has_waterfront_in_patch",
            "shoreline_raster_occupancy",
            "waterfront_raster_occupancy",
        ]
    ].copy()
    merged = rows.merge(env_only, on="sample_id", how="left", validate="one_to_one")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8", newline="") as handle:
        merged.to_csv(handle, index=False)
    return int(len(merged))


def build_tile_way_cache(
    samples: pd.DataFrame,
    output_root: Path,
    overpass_url: str,
    patch_radius_m: float,
    sleep_between_queries: float,
    max_tiles: int,
) -> dict[str, list[FeatureWay]]:
    tile_way_cache: dict[str, list[FeatureWay]] = {}
    tile_groups = list(samples.groupby("tile_id", sort=True))
    if max_tiles > 0:
        tile_groups = tile_groups[:max_tiles]
    cache_dir = output_root / "osm_cache" / "tiles"
    print(f"[osm] tiles_to_fetch={len(tile_groups)}")
    for tile_idx, (tile_id, tile_points) in enumerate(tile_groups, start=1):
        print(f"[osm] tile {tile_idx}/{len(tile_groups)} {tile_id} samples={len(tile_points)}")
        tile_way_cache[str(tile_id)] = fetch_tile_osm(
            tile_id=str(tile_id),
            tile_points=tile_points,
            cache_dir=cache_dir,
            overpass_url=overpass_url,
            patch_radius_m=patch_radius_m,
            sleep_seconds=sleep_between_queries,
        )
    missing_tile_ids = set(samples["tile_id"].astype(str)) - set(tile_way_cache)
    for tile_id in missing_tile_ids:
        tile_way_cache[tile_id] = []
    return tile_way_cache


def write_readme(output_root: Path, args: argparse.Namespace) -> None:
    text = f"""# Environment Package v1

This package adds sample-centric environment context to `clean_ship_core_lite_v1`.

Contents:

- `anchors/*.csv`: recovered anchor lat/lon for each sample at `hist_end_ts`
- `features/*/environment_features.csv`: numeric shoreline and waterfront features
- `vectors/*/vectors.jsonl.gz`: local target-centered vector features clipped to the patch
- `rasters/*/environment_rasters.npz`: compressed 2-channel raster masks
- `augmented_splits/*/part-000.csv.gz`: original sample rows plus numeric environment columns
- `summary.json`: build statistics

Patch setup:

- patch radius: {args.patch_radius_m} m
- grid size: {args.grid_size}
- tile size: {args.tile_deg} degrees
- OSM source: {args.overpass_url}

Feature channels:

- channel 0: shoreline-like features (`natural=coastline`, `waterway=riverbank|dock|canal`)
- channel 1: manmade waterfront (`pier`, `breakwater`, `groyne`, `quay`, `port_boundary`)
"""
    (output_root / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    global args
    args = parse_args()
    output_root = args.output_dir or (args.clean_root / "environment_v1")
    output_root.mkdir(parents=True, exist_ok=True)

    print("[build] collecting clean samples")
    samples = collect_clean_samples(args.clean_root, args.max_samples_per_split)
    partition_count = len(list(args.stage10_dir.glob("part-*.csv.gz")))
    samples["partition_id"] = samples["mmsi"].astype(int) % partition_count

    print("[build] recovering sample anchor lat/lon")
    samples = recover_anchor_points(samples, args.stage10_dir, args.chunk_size, args.tile_deg)
    export_anchor_csv(samples, output_root / "anchors" / "all_anchors.csv")
    for split, split_df in samples.groupby("split", sort=False):
        export_anchor_csv(split_df, output_root / "anchors" / f"{split}_anchors.csv")

    print("[build] fetching and caching OSM tiles")
    tile_way_cache = build_tile_way_cache(
        samples=samples,
        output_root=output_root,
        overpass_url=args.overpass_url,
        patch_radius_m=args.patch_radius_m,
        sleep_between_queries=args.sleep_between_queries,
        max_tiles=args.max_tiles,
    )

    summary = {
        "config": {
            "clean_root": str(args.clean_root),
            "stage10_dir": str(args.stage10_dir),
            "output_root": str(output_root),
            "overpass_urls": parse_overpass_urls(args.overpass_url),
            "patch_radius_m": args.patch_radius_m,
            "grid_size": args.grid_size,
            "tile_deg": args.tile_deg,
            "line_sample_step_m": args.line_sample_step_m,
            "max_samples_per_split": args.max_samples_per_split,
            "max_tiles": args.max_tiles,
        },
        "tiles": {
            tile_id: len(ways) for tile_id, ways in sorted(tile_way_cache.items())
        },
    }

    all_feature_frames = []
    for split in ("train", "val", "test"):
        split_samples = samples.loc[samples["split"] == split].copy().reset_index(drop=True)
        print(f"[build] split={split} samples={len(split_samples)}")
        split_summary = build_environment_for_split(
            split_samples=split_samples,
            tile_way_cache=tile_way_cache,
            feature_dir=output_root / "features" / split,
            vector_dir=output_root / "vectors" / split,
            raster_dir=output_root / "rasters" / split,
            patch_radius_m=args.patch_radius_m,
            grid_size=args.grid_size,
            line_sample_step_m=args.line_sample_step_m,
            log_every=args.log_every,
        )
        feature_dir = output_root / "features" / split
        env_features = pd.read_csv(feature_dir / "environment_features.csv")
        env_features["split"] = split
        all_feature_frames.append(env_features)
        augmented_rows = export_augmented_split(
            clean_root=args.clean_root,
            split=split,
            env_features=env_features,
            output_path=output_root / "augmented_splits" / split / "part-000.csv.gz",
        )
        split_summary["augmented_rows"] = augmented_rows
        summary[split] = split_summary

    all_features = pd.concat(all_feature_frames, ignore_index=True)
    all_features.to_csv(output_root / "all_environment_features.csv", index=False)
    write_readme(output_root, args)

    with open(output_root / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print("[build] finished")


if __name__ == "__main__":
    main()
