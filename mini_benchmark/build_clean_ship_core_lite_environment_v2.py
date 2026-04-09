#!/usr/bin/env python3
"""Build environment_v2 for clean_ship_core_lite_v1.

Minimal runnable core:
- target-centric local patches centered at anchor
- mixed vector + raster + descriptor outputs
- occupancy semantics derived from boundary masks with flood-fill from anchor
- 128x128 raster with explicit land/water/navigable channels and signed distances
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw


DEFAULT_CLEAN_ROOT = Path(
    "/mnt/nfs/kun/DeepJSCC/ship_trajectory_datesets/mini_benchmark/clean_ship_core_lite_v1"
)
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
EARTH_RADIUS_M = 6_371_000.0
GRID_CHANNELS = [
    "land_mask",
    "water_mask",
    "geo_navigable_mask",
    "natural_boundary_mask",
    "manmade_boundary_mask",
    "barrier_mask",
    "signed_dist_shore",
    "signed_dist_nav",
]


def log(message: str) -> None:
    print(message, flush=True)


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
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--source-env-root", type=Path, default=None)
    parser.add_argument("--overpass-url", type=str, default=OVERPASS_URL)
    parser.add_argument("--tile-deg", type=float, default=0.25)
    parser.add_argument("--patch-radius-m", type=float, default=5000.0)
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--line-sample-step-m", type=float, default=32.0)
    parser.add_argument("--max-samples-per-split", type=int, default=0)
    parser.add_argument("--max-tiles", type=int, default=0)
    parser.add_argument("--sleep-between-queries", type=float, default=0.5)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--skip-failed-tiles", type=lambda x: str(x).lower() not in {"false", "0", "no"}, default=True)
    parser.add_argument("--request-timeout-sec", type=float, default=90.0)
    parser.add_argument("--connect-timeout-sec", type=float, default=20.0)
    parser.add_argument("--max-fetch-attempts", type=int, default=3)
    return parser.parse_args()


def patch_margin_deg_lat(meters: float) -> float:
    return math.degrees(meters / EARTH_RADIUS_M)


def patch_margin_deg_lon(meters: float, ref_lat: float) -> float:
    denom = EARTH_RADIUS_M * max(math.cos(math.radians(ref_lat)), 1e-6)
    return math.degrees(meters / denom)


def latlon_to_local_xy(lat: np.ndarray, lon: np.ndarray, ref_lat: float, ref_lon: float) -> tuple[np.ndarray, np.ndarray]:
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    ref_lat_rad = math.radians(ref_lat)
    ref_lon_rad = math.radians(ref_lon)
    x = (lon_rad - ref_lon_rad) * math.cos(ref_lat_rad) * EARTH_RADIUS_M
    y = (lat_rad - ref_lat_rad) * EARTH_RADIUS_M
    return x, y


def safe_json_dumps(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def clip_value(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def segment_length(x0: float, y0: float, x1: float, y1: float) -> float:
    return math.hypot(x1 - x0, y1 - y0)


def point_to_segment_distance(px: float, py: float, x0: float, y0: float, x1: float, y1: float) -> float:
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


def tile_id_for_point(lat: float, lon: float, tile_deg: float) -> str:
    tile_lat = math.floor(lat / tile_deg) * tile_deg
    tile_lon = math.floor(lon / tile_deg) * tile_deg
    return f"{tile_lat:+08.3f}_{tile_lon:+09.3f}"


def make_overpass_query(south: float, west: float, north: float, east: float) -> str:
    return f"""
[out:json][timeout:240];
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
        return "natural_boundary", "coastline"
    if waterway in {"riverbank", "dock", "canal"}:
        return "natural_boundary", waterway
    if man_made in {"pier", "breakwater", "groyne", "quay"}:
        return "manmade_boundary", man_made
    if landuse == "port":
        return "manmade_boundary", "port_boundary"
    return None


def parse_osm_ways(payload: dict) -> list[FeatureWay]:
    ways = []
    for element in payload.get("elements", []):
        if element.get("type") != "way":
            continue
        classification = classify_way(element.get("tags", {}))
        if classification is None:
            continue
        geom = element.get("geometry", [])
        if len(geom) < 2:
            continue
        ways.append(
            FeatureWay(
                osm_id=int(element["id"]),
                category=classification[0],
                subtype=classification[1],
                lat=tuple(float(node["lat"]) for node in geom),
                lon=tuple(float(node["lon"]) for node in geom),
            )
        )
    return ways


def fetch_tile_osm(
    tile_id: str,
    tile_points: list[dict],
    cache_dir: Path,
    overpass_url: str,
    patch_radius_m: float,
    sleep_seconds: float,
    connect_timeout_sec: float,
    request_timeout_sec: float,
    max_fetch_attempts: int,
) -> list[FeatureWay]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{tile_id}.json"
    if cache_path.exists():
        log(f"[env_v2] cache_hit {tile_id}")
        with cache_path.open("r", encoding="utf-8") as handle:
            return parse_osm_ways(json.load(handle))

    lats = [float(row["anchor_lat"]) for row in tile_points]
    lons = [float(row["anchor_lon"]) for row in tile_points]
    south = min(lats)
    north = max(lats)
    west = min(lons)
    east = max(lons)
    center_lat = 0.5 * (south + north)
    lat_margin = patch_margin_deg_lat(patch_radius_m * 1.3)
    lon_margin = patch_margin_deg_lon(patch_radius_m * 1.3, center_lat)
    query = make_overpass_query(south - lat_margin, west - lon_margin, north + lat_margin, east + lon_margin)

    last_error = None
    for attempt in range(max(1, max_fetch_attempts)):
        try:
            log(
                f"[env_v2] fetch_start tile={tile_id} attempt={attempt + 1}/{max(1, max_fetch_attempts)} "
                f"timeout={request_timeout_sec}s"
            )
            response = requests.post(
                overpass_url,
                data=query.encode("utf-8"),
                headers={"Content-Type": "text/plain; charset=utf-8"},
                timeout=(connect_timeout_sec, request_timeout_sec),
            )
            response.raise_for_status()
            payload = response.json()
            with cache_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            log(f"[env_v2] fetch_ok tile={tile_id} ways={len(payload.get('elements', []))}")
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            return parse_osm_ways(payload)
        except Exception as exc:
            last_error = exc
            wait_sec = min(10.0, 1.5 * (attempt + 1))
            log(f"[env_v2] fetch_fail tile={tile_id} attempt={attempt + 1}: {exc}; wait={wait_sec:.1f}s")
            time.sleep(wait_sec)
    raise RuntimeError(f"Failed to fetch OSM tile {tile_id}: {last_error}")


def load_anchor_rows(source_env_root: Path, split: str, max_rows: int = 0) -> list[dict]:
    path = source_env_root / "anchors" / f"{split}_anchors.csv"
    rows = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            row["split"] = split
            row["tile_id"] = tile_id_for_point(float(row["anchor_lat"]), float(row["anchor_lon"]), 0.25)
            rows.append(row)
            if max_rows > 0 and idx + 1 >= max_rows:
                break
    return rows


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


def clip_segment_to_square(x0: float, y0: float, x1: float, y1: float, radius: float) -> tuple[float, float, float, float] | None:
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


def clip_way_to_local_segments(way: FeatureWay, anchor_lat: float, anchor_lon: float, radius_m: float) -> list[np.ndarray]:
    lat = np.asarray(way.lat, dtype=np.float64)
    lon = np.asarray(way.lon, dtype=np.float64)
    x, y = latlon_to_local_xy(lat, lon, anchor_lat, anchor_lon)
    if np.max(x) < -radius_m or np.min(x) > radius_m or np.max(y) < -radius_m or np.min(y) > radius_m:
        return []
    segments = []
    for idx in range(1, len(x)):
        clipped = clip_segment_to_square(float(x[idx - 1]), float(y[idx - 1]), float(x[idx]), float(y[idx]), radius_m)
        if clipped is None:
            continue
        segments.append(np.asarray([[clipped[0], clipped[1]], [clipped[2], clipped[3]]], dtype=np.float32))
    return segments


def metric_to_grid(x: np.ndarray, y: np.ndarray, radius_m: float, grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    gx = np.floor((x + radius_m) / (2.0 * radius_m) * grid_size).astype(np.int32)
    gy = np.floor((radius_m - y) / (2.0 * radius_m) * grid_size).astype(np.int32)
    return gx, gy


def rasterize_segments(segments: list[np.ndarray], radius_m: float, grid_size: int, sample_step_m: float) -> np.ndarray:
    mask = np.zeros((grid_size, grid_size), dtype=np.uint8)
    if not segments:
        return mask
    for segment in segments:
        x0, y0 = map(float, segment[0])
        x1, y1 = map(float, segment[1])
        length = max(segment_length(x0, y0, x1, y1), 1.0)
        num = max(2, int(math.ceil(length / max(sample_step_m, 1.0))) + 1)
        xs = np.linspace(x0, x1, num=num)
        ys = np.linspace(y0, y1, num=num)
        gx, gy = metric_to_grid(xs, ys, radius_m, grid_size)
        valid = (gx >= 0) & (gx < grid_size) & (gy >= 0) & (gy < grid_size)
        mask[gy[valid], gx[valid]] = 1
    return mask


def binary_dilate(mask: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return mask.copy()
    out = mask.copy()
    for dy in range(-radius_px, radius_px + 1):
        for dx in range(-radius_px, radius_px + 1):
            if dx * dx + dy * dy > radius_px * radius_px:
                continue
            src_y0 = max(0, -dy)
            src_y1 = min(mask.shape[0], mask.shape[0] - dy)
            src_x0 = max(0, -dx)
            src_x1 = min(mask.shape[1], mask.shape[1] - dx)
            dst_y0 = max(0, dy)
            dst_y1 = min(mask.shape[0], mask.shape[0] + dy)
            dst_x0 = max(0, dx)
            dst_x1 = min(mask.shape[1], mask.shape[1] + dx)
            out[dst_y0:dst_y1, dst_x0:dst_x1] |= mask[src_y0:src_y1, src_x0:src_x1]
    return out


def flood_fill_water(barrier_mask: np.ndarray) -> np.ndarray:
    h, w = barrier_mask.shape
    water = np.zeros_like(barrier_mask, dtype=np.uint8)
    cy = h // 2
    cx = w // 2
    if barrier_mask[cy, cx] > 0:
        found = False
        for radius in range(1, 5):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    yy = cy + dy
                    xx = cx + dx
                    if 0 <= yy < h and 0 <= xx < w and barrier_mask[yy, xx] == 0:
                        cy, cx = yy, xx
                        found = True
                        break
                if found:
                    break
            if found:
                break
    if barrier_mask[cy, cx] > 0:
        return water
    queue = [(cy, cx)]
    water[cy, cx] = 1
    head = 0
    while head < len(queue):
        y, x = queue[head]
        head += 1
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                continue
            if water[ny, nx] or barrier_mask[ny, nx]:
                continue
            water[ny, nx] = 1
            queue.append((ny, nx))
    return water


def unsigned_distance_map(mask: np.ndarray, radius_m: float) -> np.ndarray:
    h, w = mask.shape
    ys, xs = np.nonzero(mask > 0)
    cell_size = (2.0 * radius_m) / max(h, 1)
    if len(xs) == 0:
        return np.full((h, w), radius_m, dtype=np.float32)
    grid_y, grid_x = np.indices((h, w), dtype=np.float32)
    points = np.stack([ys.astype(np.float32), xs.astype(np.float32)], axis=1)
    out = np.full((h, w), np.inf, dtype=np.float32)
    chunk = 512
    flat_y = grid_y.reshape(-1)
    flat_x = grid_x.reshape(-1)
    flat_out = np.full(flat_y.shape[0], np.inf, dtype=np.float32)
    for start in range(0, len(points), chunk):
        part = points[start : start + chunk]
        dy = flat_y[:, None] - part[None, :, 0]
        dx = flat_x[:, None] - part[None, :, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        flat_out = np.minimum(flat_out, dist.min(axis=1))
    out = flat_out.reshape(h, w) * cell_size
    return out.astype(np.float32)


def signed_distance(reference_mask: np.ndarray, positive_mask: np.ndarray, radius_m: float) -> np.ndarray:
    dist = unsigned_distance_map(reference_mask, radius_m)
    sign = np.where(positive_mask > 0, 1.0, -1.0)
    return (dist * sign).astype(np.float32)


def summarize_segments(segments: list[np.ndarray]) -> tuple[float, float]:
    if not segments:
        return 0.0, 0.0
    total_length = 0.0
    min_dist = float("inf")
    for segment in segments:
        x0, y0 = map(float, segment[0])
        x1, y1 = map(float, segment[1])
        total_length += segment_length(x0, y0, x1, y1)
        min_dist = min(min_dist, point_to_segment_distance(0.0, 0.0, x0, y0, x1, y1))
    return total_length, (min_dist if math.isfinite(min_dist) else 0.0)


def build_scene_type(water_ratio: float, navigable_ratio: float, nearest_shore_dist_m: float, manmade_density: float) -> tuple[str, dict[str, int]]:
    if manmade_density > 0.02 and nearest_shore_dist_m < 500.0:
        scene = "harbor"
    elif water_ratio < 0.45 or navigable_ratio < 0.3:
        scene = "constrained"
    elif nearest_shore_dist_m < 1200.0:
        scene = "nearshore"
    else:
        scene = "open_water"
    return scene, {
        "scene_open_water": int(scene == "open_water"),
        "scene_nearshore": int(scene == "nearshore"),
        "scene_harbor": int(scene == "harbor"),
        "scene_constrained": int(scene == "constrained"),
    }


def make_quality_score(anchor_in_water: int, anchor_in_nav: int, water_ratio: float, navigable_ratio: float) -> float:
    score = 0.25 * float(anchor_in_water) + 0.35 * float(anchor_in_nav)
    score += 0.2 * min(max(water_ratio, 0.0), 1.0)
    score += 0.2 * min(max(navigable_ratio, 0.0), 1.0)
    return float(min(max(score, 0.0), 1.0))


def build_sample_environment(row: dict, ways: list[FeatureWay], patch_radius_m: float, grid_size: int, sample_step_m: float) -> tuple[dict, np.ndarray, dict]:
    natural_segments = []
    manmade_segments = []
    vector_payload = {
        "sample_id": row["sample_id"],
        "split": row["split"],
        "anchor_lat": float(row["anchor_lat"]),
        "anchor_lon": float(row["anchor_lon"]),
        "tile_id": row["tile_id"],
        "transform": {
            "patch_radius_m": patch_radius_m,
            "grid_size": grid_size,
            "meters_per_pixel": (2.0 * patch_radius_m) / grid_size,
        },
        "natural_boundary": [],
        "manmade_boundary": [],
        "barrier": [],
    }

    for way in ways:
        clipped_segments = clip_way_to_local_segments(
            way,
            anchor_lat=float(row["anchor_lat"]),
            anchor_lon=float(row["anchor_lon"]),
            radius_m=patch_radius_m,
        )
        if not clipped_segments:
            continue
        target = natural_segments if way.category == "natural_boundary" else manmade_segments
        target.extend(clipped_segments)
        key = "natural_boundary" if way.category == "natural_boundary" else "manmade_boundary"
        for segment in clipped_segments:
            rounded = [[round(float(pt[0]), 2), round(float(pt[1]), 2)] for pt in segment.tolist()]
            payload = {"osm_id": int(way.osm_id), "subtype": str(way.subtype), "xy": rounded}
            vector_payload[key].append(payload)
            vector_payload["barrier"].append(payload)

    natural_mask = rasterize_segments(natural_segments, patch_radius_m, grid_size, sample_step_m)
    manmade_mask = rasterize_segments(manmade_segments, patch_radius_m, grid_size, sample_step_m)
    barrier_mask = np.maximum(natural_mask, manmade_mask)
    water_mask = flood_fill_water(barrier_mask)
    land_mask = (1 - water_mask).astype(np.uint8)
    non_nav = binary_dilate(barrier_mask, 1)
    geo_nav = (water_mask > 0).astype(np.uint8)
    geo_nav[non_nav > 0] = 0
    if geo_nav[grid_size // 2, grid_size // 2] == 0 and water_mask[grid_size // 2, grid_size // 2] > 0:
        geo_nav[grid_size // 2, grid_size // 2] = 1

    signed_shore = signed_distance(barrier_mask, water_mask, patch_radius_m)
    signed_nav = signed_distance(1 - geo_nav, geo_nav, patch_radius_m)

    total_cells = float(grid_size * grid_size)
    natural_length, nearest_natural = summarize_segments(natural_segments)
    manmade_length, nearest_manmade = summarize_segments(manmade_segments)
    nearest_barrier = min(nearest_natural if natural_segments else patch_radius_m, nearest_manmade if manmade_segments else patch_radius_m)
    water_ratio = float(water_mask.sum() / total_cells)
    navigable_ratio = float(geo_nav.sum() / total_cells)
    natural_density = float(natural_mask.sum() / total_cells)
    manmade_density = float(manmade_mask.sum() / total_cells)
    barrier_density = float(barrier_mask.sum() / total_cells)
    scene_name, scene_flags = build_scene_type(water_ratio, navigable_ratio, nearest_natural if natural_segments else patch_radius_m, manmade_density)
    anchor_in_water = int(water_mask[grid_size // 2, grid_size // 2] > 0)
    anchor_in_nav = int(geo_nav[grid_size // 2, grid_size // 2] > 0)
    anchor_on_barrier = int(barrier_mask[grid_size // 2, grid_size // 2] > 0)
    quality_score = make_quality_score(anchor_in_water, anchor_in_nav, water_ratio, navigable_ratio)

    descriptors = {
        "sample_id": row["sample_id"],
        "split": row["split"],
        "segment_id": row["segment_id"],
        "hist_end_ts": row["hist_end_ts"],
        "anchor_lat": float(row["anchor_lat"]),
        "anchor_lon": float(row["anchor_lon"]),
        "tile_id": row["tile_id"],
        "patch_radius_m": patch_radius_m,
        "grid_size": grid_size,
        "nearest_shore_dist_m": round(float(nearest_natural if natural_segments else patch_radius_m), 3),
        "nearest_manmade_dist_m": round(float(nearest_manmade if manmade_segments else patch_radius_m), 3),
        "nearest_barrier_dist_m": round(float(nearest_barrier), 3),
        "water_ratio": round(water_ratio, 6),
        "navigable_ratio": round(navigable_ratio, 6),
        "boundary_density": round(barrier_density, 6),
        "natural_boundary_density": round(natural_density, 6),
        "manmade_boundary_density": round(manmade_density, 6),
        "barrier_density": round(barrier_density, 6),
        "natural_boundary_length_m": round(natural_length, 3),
        "manmade_boundary_length_m": round(manmade_length, 3),
        "has_natural_boundary": int(bool(natural_segments)),
        "has_manmade_boundary": int(bool(manmade_segments)),
        "anchor_in_water": anchor_in_water,
        "anchor_in_navigable": anchor_in_nav,
        "anchor_on_barrier": anchor_on_barrier,
        "scene_type": scene_name,
        "env_quality_score": round(quality_score, 6),
    }
    descriptors.update(scene_flags)

    raster = np.zeros((8, grid_size, grid_size), dtype=np.float32)
    raster[0] = land_mask
    raster[1] = water_mask
    raster[2] = geo_nav
    raster[3] = natural_mask
    raster[4] = manmade_mask
    raster[5] = barrier_mask
    raster[6] = signed_shore
    raster[7] = signed_nav
    quality = {
        "anchor_in_water": anchor_in_water,
        "anchor_in_navigable": anchor_in_nav,
        "anchor_on_barrier": anchor_on_barrier,
        "scene_type": scene_name,
        "env_quality_score": quality_score,
    }
    merged_descriptors = dict(descriptors)
    merged_descriptors.update(quality)
    return vector_payload, raster, merged_descriptors, quality


def export_augmented_split(clean_root: Path, split: str, desc_rows: list[dict], output_path: Path) -> int:
    source_path = clean_root / split / "part-000.csv.gz"
    desc_index = {row["sample_id"]: row for row in desc_rows}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(source_path, "rt", encoding="utf-8") as source, gzip.open(output_path, "wt", encoding="utf-8", newline="") as target:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames + [column for column in desc_rows[0].keys() if column not in reader.fieldnames]
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        count = 0
        for row in reader:
            row.update(desc_index.get(row["sample_id"], {}))
            writer.writerow(row)
            count += 1
    return count


def write_readme(output_root: Path, args: argparse.Namespace) -> None:
    text = f"""# Environment Package v2

Mixed sample-centric environment context for `clean_ship_core_lite_v1`.

Contents:

- `anchors/*.csv`: anchor lat/lon metadata copied from v1
- `features/*/environment_descriptors.csv`: numeric descriptors and quality flags
- `vectors/*/vectors.jsonl.gz`: target-centered local vector objects
- `rasters/*/masks.npy`: raster masks for occupancy and boundaries
- `rasters/*/signed_dist_shore.npy`: signed distance to shoreline/barrier semantics
- `rasters/*/signed_dist_nav.npy`: signed distance to navigable region
- `rasters/*/sample_ids.npy`: aligned sample ids for raster arrays
- `augmented_splits/*/part-000.csv.gz`: original sample rows plus environment descriptors
- `summary.json`: build statistics

Patch setup:

- patch radius: {args.patch_radius_m} m
- grid size: {args.grid_size}
- tile size: {args.tile_deg} degrees
- OSM source: {args.overpass_url}

Raster channels:

1. `land_mask`
2. `water_mask`
3. `geo_navigable_mask`
4. `natural_boundary_mask`
5. `manmade_boundary_mask`
6. `barrier_mask`
7. `signed_dist_shore`
8. `signed_dist_nav`
"""
    (output_root / "README.md").write_text(text, encoding="utf-8")


def compute_descriptor_stats(all_rows: list[dict]) -> dict:
    max_log = {}
    for column in ("boundary_density", "natural_boundary_density", "manmade_boundary_density", "barrier_density"):
        values = [float(row.get(column, 0.0)) for row in all_rows]
        max_log[column] = float(max(np.log1p(np.maximum(values, 0.0))) if values else 1.0)
    return {"log1p_max": max_log}


def build_split(rows: list[dict], ways_by_tile: dict[str, list[FeatureWay]], output_root: Path, args: argparse.Namespace) -> tuple[list[dict], dict]:
    split = rows[0]["split"] if rows else "unknown"
    vector_dir = output_root / "vectors" / split
    raster_dir = output_root / "rasters" / split
    feature_dir = output_root / "features" / split
    vector_dir.mkdir(parents=True, exist_ok=True)
    raster_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)

    desc_rows = []
    qualities = []
    sample_ids = []
    masks = np.zeros((len(rows), 6, args.grid_size, args.grid_size), dtype=np.uint8)
    signed_dist_shore = np.zeros((len(rows), args.grid_size, args.grid_size), dtype=np.float16)
    signed_dist_nav = np.zeros((len(rows), args.grid_size, args.grid_size), dtype=np.float16)

    with gzip.open(vector_dir / "vectors.jsonl.gz", "wt", encoding="utf-8") as vector_handle:
        for idx, row in enumerate(rows):
            vector_payload, raster, descriptors, quality = build_sample_environment(
                row,
                ways_by_tile.get(row["tile_id"], []),
                patch_radius_m=args.patch_radius_m,
                grid_size=args.grid_size,
                sample_step_m=args.line_sample_step_m,
            )
            vector_handle.write(safe_json_dumps(vector_payload) + "\n")
            desc_rows.append(descriptors)
            qualities.append(quality)
            sample_ids.append(row["sample_id"])
            masks[idx] = raster[:6].astype(np.uint8)
            signed_dist_shore[idx] = raster[6].astype(np.float16)
            signed_dist_nav[idx] = raster[7].astype(np.float16)
            if (idx + 1) % args.log_every == 0:
                log(f"[env_v2] split={split} processed={idx + 1}/{len(rows)}")

    with (feature_dir / "environment_descriptors.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(desc_rows[0].keys()))
        writer.writeheader()
        for row in desc_rows:
            writer.writerow(row)

    np.save(raster_dir / "sample_ids.npy", np.asarray(sample_ids, dtype=object), allow_pickle=True)
    np.save(raster_dir / "masks.npy", masks, allow_pickle=False)
    np.save(raster_dir / "signed_dist_shore.npy", signed_dist_shore, allow_pickle=False)
    np.save(raster_dir / "signed_dist_nav.npy", signed_dist_nav, allow_pickle=False)

    summary = {
        "num_samples": len(rows),
        "anchor_in_water": int(sum(item["anchor_in_water"] for item in qualities)),
        "anchor_in_navigable": int(sum(item["anchor_in_navigable"] for item in qualities)),
        "mean_quality": float(np.mean([item["env_quality_score"] for item in qualities])) if qualities else 0.0,
    }
    return desc_rows, summary


def build_tile_way_cache(all_rows: list[dict], output_root: Path, args: argparse.Namespace) -> tuple[dict[str, list[FeatureWay]], list[dict]]:
    groups = defaultdict(list)
    for row in all_rows:
        groups[row["tile_id"]].append(row)
    tile_items = sorted(groups.items())
    if args.max_tiles > 0:
        tile_items = tile_items[: args.max_tiles]
    cache = {}
    cache_dir = output_root / "osm_cache" / "tiles"
    failures = []
    progress_path = output_root / "tile_progress.json"
    failed_partial_path = output_root / "failed_tiles.partial.json"
    log(f"[env_v2] tiles_to_fetch={len(tile_items)}")
    for idx, (tile_id, tile_rows) in enumerate(tile_items, start=1):
        log(f"[env_v2] tile {idx}/{len(tile_items)} {tile_id} samples={len(tile_rows)}")
        try:
            cache[tile_id] = fetch_tile_osm(
                tile_id=tile_id,
                tile_points=tile_rows,
                cache_dir=cache_dir,
                overpass_url=args.overpass_url,
                patch_radius_m=args.patch_radius_m,
                sleep_seconds=args.sleep_between_queries,
                connect_timeout_sec=args.connect_timeout_sec,
                request_timeout_sec=args.request_timeout_sec,
                max_fetch_attempts=args.max_fetch_attempts,
            )
        except Exception as exc:
            failure = {
                "tile_id": tile_id,
                "sample_count": len(tile_rows),
                "error": str(exc),
            }
            failures.append(failure)
            log(f"[env_v2] tile_failed {tile_id}: {exc}")
            if args.skip_failed_tiles:
                cache[tile_id] = []
            else:
                raise
        with progress_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "done_tiles": idx,
                    "total_tiles": len(tile_items),
                    "last_tile_id": tile_id,
                    "failed_tiles": len(failures),
                },
                handle,
                indent=2,
            )
        with failed_partial_path.open("w", encoding="utf-8") as handle:
            json.dump(failures, handle, indent=2)
    missing = set(groups) - set(cache)
    for tile_id in missing:
        cache[tile_id] = []
    return cache, failures


def copy_anchor_csvs(source_env_root: Path, output_root: Path) -> None:
    target_dir = output_root / "anchors"
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in ("all_anchors.csv", "train_anchors.csv", "val_anchors.csv", "test_anchors.csv"):
        src = source_env_root / "anchors" / name
        dst = target_dir / name
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_env_root = args.source_env_root or (args.clean_root / "environment_v1")
    output_root = args.output_dir or (args.clean_root / "environment_v2")
    output_root.mkdir(parents=True, exist_ok=True)

    all_rows = []
    split_rows = {}
    for split in ("train", "val", "test"):
        rows = load_anchor_rows(source_env_root, split, args.max_samples_per_split)
        split_rows[split] = rows
        all_rows.extend(rows)

    copy_anchor_csvs(source_env_root, output_root)
    ways_by_tile, failed_tiles = build_tile_way_cache(all_rows, output_root, args)

    summary = {
        "config": {
            "clean_root": str(args.clean_root),
            "output_root": str(output_root),
            "patch_radius_m": args.patch_radius_m,
            "grid_size": args.grid_size,
            "tile_deg": args.tile_deg,
            "line_sample_step_m": args.line_sample_step_m,
            "max_samples_per_split": args.max_samples_per_split,
            "max_tiles": args.max_tiles,
        },
        "tiles": {tile_id: len(ways) for tile_id, ways in sorted(ways_by_tile.items())},
        "failed_tiles": failed_tiles,
    }

    all_desc = []
    for split in ("train", "val", "test"):
        log(f"[env_v2] split={split} samples={len(split_rows[split])}")
        desc_rows, split_summary = build_split(split_rows[split], ways_by_tile, output_root, args)
        if desc_rows:
            export_augmented_split(
                args.clean_root,
                split,
                desc_rows,
                output_root / "augmented_splits" / split / "part-000.csv.gz",
            )
            all_desc.extend(desc_rows)
        summary[split] = split_summary

    if all_desc:
        with (output_root / "all_environment_descriptors.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_desc[0].keys()))
            writer.writeheader()
            for row in all_desc:
                writer.writerow(row)
        with (output_root / "feature_stats_v2.json").open("w", encoding="utf-8") as handle:
            json.dump(compute_descriptor_stats(all_desc), handle, indent=2, sort_keys=True)

    write_readme(output_root, args)
    with (output_root / "failed_tiles.json").open("w", encoding="utf-8") as handle:
        json.dump(failed_tiles, handle, indent=2)
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    log(f"[env_v2] finished failed_tiles={len(failed_tiles)}")


if __name__ == "__main__":
    main()
