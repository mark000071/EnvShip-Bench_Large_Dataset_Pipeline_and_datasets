#!/usr/bin/env python3
"""Assign scene groups and regional clusters for clean_ship_core_lite_v1."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_CLEAN_ROOT = Path(
    "/mnt/nfs/kun/DeepJSCC/NOAA_ship_trajectory_datasets/mini_bench/clean_ship_core_lite_v1"
)
DEFAULT_ENV_ROOT = DEFAULT_CLEAN_ROOT / "environment_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-root", type=Path, default=DEFAULT_CLEAN_ROOT)
    parser.add_argument("--env-root", type=Path, default=DEFAULT_ENV_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ENV_ROOT / "scene_groups_v1")
    return parser.parse_args()


def parse_tile_lat_lon(tile_id: str) -> tuple[float, float]:
    return float(tile_id[:8]), float(tile_id[9:])


def load_all_features(env_root: Path) -> pd.DataFrame:
    frames = []
    for split in ("train", "val", "test"):
        frame = pd.read_csv(env_root / "features" / split / "environment_features.csv")
        frame["split"] = split
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    tile_coords = df["tile_id"].astype(str).apply(parse_tile_lat_lon)
    df["tile_lat"] = [x[0] for x in tile_coords]
    df["tile_lon"] = [x[1] for x in tile_coords]
    return df


def load_quality_metadata(clean_root: Path) -> pd.DataFrame:
    frames = []
    for split in ("train", "val", "test"):
        frame = pd.read_csv(clean_root / "reports" / f"{split}_selected_metadata.csv")
        frame["split"] = split
        frames.append(frame[["sample_id", "split", "avg_speed_knots", "ship_group"]])
    return pd.concat(frames, ignore_index=True)


def compute_tile_counts(df: pd.DataFrame) -> pd.Series:
    return df.groupby("tile_id").size().rename("tile_sample_count")


def assign_scene_group(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    tile_counts = compute_tile_counts(df)
    df["tile_sample_count"] = df["tile_id"].map(tile_counts).astype(int)

    harbor = (
        (df["min_waterfront_dist_m"] <= 1200.0)
        | (df["waterfront_segment_count"] >= 20)
        | (df["waterfront_total_length_m"] >= 1500.0)
    )
    nearshore = (
        (df["min_shoreline_dist_m"] <= 900.0)
        | (df["shoreline_segment_count"] >= 80)
        | (df["shoreline_total_length_m"] >= 2500.0)
    )
    coastal = (
        (df["min_shoreline_dist_m"] <= 2200.0)
        | (df["shoreline_segment_count"] > 0)
        | (df["min_waterfront_dist_m"] <= 2200.0)
    )

    scene_group = np.full(len(df), "offshore_route", dtype=object)
    scene_group[coastal.to_numpy()] = "coastal_route"
    scene_group[nearshore.to_numpy()] = "nearshore_channel"
    scene_group[harbor.to_numpy()] = "harbor_port"
    df["scene_group"] = scene_group

    density_group = np.full(len(df), "sparse_region", dtype=object)
    density_group[df["tile_sample_count"] >= 300] = "traffic_hub"
    density_group[(df["tile_sample_count"] >= 80) & (df["tile_sample_count"] < 300)] = "traffic_corridor"
    df["density_group"] = density_group
    return df


def build_region_clusters(df: pd.DataFrame) -> dict[str, str]:
    tiles = sorted(df["tile_id"].astype(str).unique())
    tile_set = set(tiles)
    coord_map = {tile: parse_tile_lat_lon(tile) for tile in tiles}
    tile_to_grid = {tile: (round(lat * 4), round(lon * 4)) for tile, (lat, lon) in coord_map.items()}
    grid_to_tile = {grid: tile for tile, grid in tile_to_grid.items()}

    tile_to_cluster: dict[str, str] = {}
    cluster_idx = 0
    for tile in tiles:
        if tile in tile_to_cluster:
            continue
        cluster_name = f"region_{cluster_idx:03d}"
        cluster_idx += 1
        queue = deque([tile])
        tile_to_cluster[tile] = cluster_name
        while queue:
            current = queue.popleft()
            gy, gx = tile_to_grid[current]
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    neighbor = grid_to_tile.get((gy + dy, gx + dx))
                    if neighbor is None or neighbor in tile_to_cluster:
                        continue
                    tile_to_cluster[neighbor] = cluster_name
                    queue.append(neighbor)
    return tile_to_cluster


def summarize_regions(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for region_id, group in df.groupby("region_cluster", sort=True):
        rows.append(
            {
                "region_cluster": region_id,
                "num_samples": int(len(group)),
                "num_tiles": int(group["tile_id"].nunique()),
                "lat_min": float(group["anchor_lat"].min()),
                "lat_max": float(group["anchor_lat"].max()),
                "lon_min": float(group["anchor_lon"].min()),
                "lon_max": float(group["anchor_lon"].max()),
                "dominant_scene_group": group["scene_group"].mode().iloc[0],
                "scene_group_counts": json.dumps(group["scene_group"].value_counts().to_dict(), ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows).sort_values("num_samples", ascending=False)


def export_augmented_split(clean_root: Path, split: str, scene_df: pd.DataFrame, output_path: Path) -> None:
    base = pd.read_csv(clean_root / split / "part-000.csv.gz", compression="gzip", low_memory=False)
    add_cols = scene_df.loc[scene_df["split"] == split, ["sample_id", "scene_group", "density_group", "region_cluster", "tile_sample_count"]]
    merged = base.merge(add_cols, on="sample_id", how="left", validate="one_to_one")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8", newline="") as handle:
        merged.to_csv(handle, index=False)


def plot_scene_map(df: pd.DataFrame, output_path: Path) -> None:
    colors = {
        "harbor_port": "#d62728",
        "nearshore_channel": "#ff7f0e",
        "coastal_route": "#1f77b4",
        "offshore_route": "#2ca02c",
    }
    fig, ax = plt.subplots(figsize=(9, 7))
    sample = df.sample(min(8000, len(df)), random_state=0)
    for group_name, part in sample.groupby("scene_group"):
        ax.scatter(part["anchor_lon"], part["anchor_lat"], s=5, alpha=0.45, label=group_name, color=colors[group_name], edgecolors="none")
    ax.set_title("Scene Group Geographic Distribution")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_scene_counts(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    count_df = (
        df.groupby(["split", "scene_group"])
        .size()
        .rename("count")
        .reset_index()
        .pivot(index="split", columns="scene_group", values="count")
        .fillna(0)
    )
    count_df = count_df[[c for c in ["harbor_port", "nearshore_channel", "coastal_route", "offshore_route"] if c in count_df.columns]]
    count_df.plot(kind="bar", stacked=True, ax=axes[0], color=["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"])
    axes[0].set_title("Scene Group Counts by Split")
    axes[0].set_xlabel("split")
    axes[0].set_ylabel("samples")
    axes[0].grid(True, axis="y", alpha=0.25)

    region_sizes = df.groupby("region_cluster").size().sort_values(ascending=False).head(20)
    axes[1].bar(np.arange(len(region_sizes)), region_sizes.values, color="#9467bd")
    axes[1].set_title("Top Region Clusters by Sample Count")
    axes[1].set_xlabel("region cluster rank")
    axes[1].set_ylabel("samples")
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(df: pd.DataFrame, region_summary: pd.DataFrame, output_path: Path) -> None:
    payload = {
        "num_samples": int(len(df)),
        "scene_group_counts": df["scene_group"].value_counts().to_dict(),
        "density_group_counts": df["density_group"].value_counts().to_dict(),
        "num_region_clusters": int(df["region_cluster"].nunique()),
        "top_regions": region_summary.head(20).to_dict(orient="records"),
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir = args.output_dir / "visualizations"
    visual_dir.mkdir(parents=True, exist_ok=True)

    env_df = load_all_features(args.env_root)
    quality_df = load_quality_metadata(args.clean_root)
    df = env_df.merge(quality_df, on=["sample_id", "split"], how="left", validate="one_to_one")
    df = assign_scene_group(df)
    tile_to_cluster = build_region_clusters(df)
    df["region_cluster"] = df["tile_id"].map(tile_to_cluster)

    df.to_csv(args.output_dir / "all_scene_groups.csv", index=False)
    for split in ("train", "val", "test"):
        split_df = df.loc[df["split"] == split].copy()
        split_df.to_csv(args.output_dir / f"{split}_scene_groups.csv", index=False)
        export_augmented_split(args.clean_root, split, df, args.output_dir / "augmented_splits" / split / "part-000.csv.gz")

    region_summary = summarize_regions(df)
    region_summary.to_csv(args.output_dir / "region_cluster_summary.csv", index=False)
    plot_scene_map(df, visual_dir / "scene_group_map.png")
    plot_scene_counts(df, visual_dir / "scene_group_summary.png")
    write_summary(df, region_summary, args.output_dir / "summary.json")
    print(json.dumps({"scene_group_counts": df["scene_group"].value_counts().to_dict(), "num_region_clusters": int(df["region_cluster"].nunique())}, indent=2))


if __name__ == "__main__":
    main()
