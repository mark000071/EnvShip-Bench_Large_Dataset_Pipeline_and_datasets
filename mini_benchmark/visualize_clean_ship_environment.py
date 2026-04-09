#!/usr/bin/env python3
"""Visualize environment augmentation results for clean_ship_core_lite_v1."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_CLEAN_ROOT = Path(
    "/mnt/nfs/kun/DeepJSCC/ship_trajectory_datesets/mini_benchmark/clean_ship_core_lite_v1"
)
DEFAULT_ENV_ROOT = DEFAULT_CLEAN_ROOT / "environment_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-root", type=Path, default=DEFAULT_CLEAN_ROOT)
    parser.add_argument("--env-root", type=Path, default=DEFAULT_ENV_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ENV_ROOT / "visualizations")
    parser.add_argument("--gallery-per-split", type=int, default=4)
    return parser.parse_args()


def load_features(env_root: Path, split: str) -> pd.DataFrame:
    return pd.read_csv(env_root / "features" / split / "environment_features.csv")


def select_example_ids(env_root: Path) -> dict[str, str]:
    selected: dict[str, str] = {}
    for split in ("train", "val", "test"):
        df = load_features(env_root, split)
        positive = df.loc[(df["has_shoreline_in_patch"] > 0) | (df["has_waterfront_in_patch"] > 0)].copy()
        if positive.empty:
            positive = df.copy()
        positive = positive.sort_values(
            [
                "has_waterfront_in_patch",
                "has_shoreline_in_patch",
                "shoreline_segment_count",
                "waterfront_segment_count",
            ],
            ascending=False,
        )
        selected[split] = str(positive.iloc[0]["sample_id"])
    return selected


def select_gallery_examples(env_root: Path, per_split: int) -> dict[str, list[str]]:
    chosen: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        df = load_features(env_root, split)
        ranked = df.sort_values(
            [
                "has_waterfront_in_patch",
                "has_shoreline_in_patch",
                "shoreline_segment_count",
                "waterfront_segment_count",
            ],
            ascending=False,
        )
        chosen[split] = ranked["sample_id"].astype(str).head(per_split).tolist()
    return chosen


def load_sample_row(clean_root: Path, split: str, sample_id: str) -> dict[str, str]:
    path = clean_root / split / "part-000.csv.gz"
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["sample_id"] == sample_id:
                return row
    raise KeyError(f"sample_id {sample_id} not found in {path}")


def load_vector_payload(env_root: Path, split: str, sample_id: str) -> dict:
    path = env_root / "vectors" / split / "vectors.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if payload["sample_id"] == sample_id:
                return payload
    raise KeyError(f"sample_id {sample_id} not found in {path}")


def load_raster(env_root: Path, split: str, sample_id: str) -> np.ndarray:
    path = env_root / "rasters" / split / "environment_rasters.npz"
    data = np.load(path, allow_pickle=True)
    sample_ids = data["sample_ids"].astype(str)
    matches = np.where(sample_ids == sample_id)[0]
    if len(matches) == 0:
        raise KeyError(f"sample_id {sample_id} not in raster file {path}")
    return data["raster"][int(matches[0])]


def json_array(value: str) -> np.ndarray:
    return np.asarray(json.loads(value), dtype=np.float32)


def plot_summary(env_root: Path, output_path: Path) -> None:
    stats = []
    for split in ("train", "val", "test"):
        df = load_features(env_root, split)
        stats.append(
            {
                "split": split,
                "num_samples": len(df),
                "shoreline_positive": int((df["has_shoreline_in_patch"] > 0).sum()),
                "waterfront_positive": int((df["has_waterfront_in_patch"] > 0).sum()),
                "median_min_shoreline_dist_m": float(df["min_shoreline_dist_m"].median()),
            }
        )
    stats_df = pd.DataFrame(stats)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(stats_df))
    axes[0].bar(x - 0.15, stats_df["shoreline_positive"], width=0.3, label="shoreline positive", color="#1f77b4")
    axes[0].bar(x + 0.15, stats_df["waterfront_positive"], width=0.3, label="waterfront positive", color="#ff7f0e")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(stats_df["split"].tolist())
    axes[0].set_ylabel("Sample count")
    axes[0].set_title("Positive Environment Samples")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.25)

    all_features = pd.concat([load_features(env_root, s).assign(split=s) for s in ("train", "val", "test")], ignore_index=True)
    clipped = all_features["min_shoreline_dist_m"].clip(upper=4096.0)
    axes[1].hist(clipped, bins=30, color="#2ca02c", alpha=0.85)
    axes[1].set_title("Min Shoreline Distance Distribution")
    axes[1].set_xlabel("meters (clipped at 4096)")
    axes[1].set_ylabel("Samples")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_anchor_map(env_root: Path, output_path: Path) -> None:
    anchors = pd.read_csv(env_root / "anchors" / "all_anchors.csv")
    counts = anchors.groupby("tile_id").size().rename("count").reset_index()
    counts["tile_lat"] = counts["tile_id"].str.slice(0, 8).astype(float)
    counts["tile_lon"] = counts["tile_id"].str.slice(9).astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    sample = anchors.sample(min(6000, len(anchors)), random_state=0)
    axes[0].scatter(
        sample["anchor_lon"],
        sample["anchor_lat"],
        s=4,
        alpha=0.35,
        color="#1f77b4",
        edgecolors="none",
    )
    axes[0].set_title("Anchor Distribution")
    axes[0].set_xlabel("longitude")
    axes[0].set_ylabel("latitude")
    axes[0].grid(True, alpha=0.25)

    sc = axes[1].scatter(
        counts["tile_lon"],
        counts["tile_lat"],
        c=counts["count"],
        s=np.clip(counts["count"] / 8.0, 10, 180),
        cmap="viridis",
        alpha=0.9,
    )
    axes[1].set_title("Tile-Level Sample Density")
    axes[1].set_xlabel("tile longitude")
    axes[1].set_ylabel("tile latitude")
    axes[1].grid(True, alpha=0.25)
    fig.colorbar(sc, ax=axes[1], label="samples per tile")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_example(
    clean_root: Path,
    env_root: Path,
    split: str,
    sample_id: str,
    output_path: Path,
) -> None:
    row = load_sample_row(clean_root, split, sample_id)
    payload = load_vector_payload(env_root, split, sample_id)
    raster = load_raster(env_root, split, sample_id)
    hist_x = json_array(row["hist_x_json"])
    hist_y = json_array(row["hist_y_json"])
    fut_x = json_array(row["fut_x_json"])
    fut_y = json_array(row["fut_y_json"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    ax = axes[0]
    for item in payload["shoreline"]:
        xy = np.asarray(item["xy"], dtype=np.float32)
        ax.plot(xy[:, 0], xy[:, 1], color="#1f77b4", linewidth=1.0, alpha=0.9)
    for item in payload["waterfront"]:
        xy = np.asarray(item["xy"], dtype=np.float32)
        ax.plot(xy[:, 0], xy[:, 1], color="#ff7f0e", linewidth=1.0, alpha=0.9)
    ax.plot(hist_x, hist_y, "-o", color="#2ca02c", markersize=2.5, linewidth=1.4, label="history")
    ax.plot(fut_x, fut_y, "-o", color="#d62728", markersize=2.5, linewidth=1.4, label="future")
    ax.scatter([0.0], [0.0], color="black", s=28, zorder=5, label="anchor")
    ax.set_title(f"{split}: vector environment\n{sample_id}")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax = axes[1]
    combined = np.zeros((raster.shape[1], raster.shape[2], 3), dtype=np.float32)
    combined[..., 2] = raster[0]  # shoreline blue
    combined[..., 0] = raster[1]  # waterfront red
    ax.imshow(combined, origin="lower")
    ax.set_title("Raster channels\nblue=shoreline red=waterfront")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_gallery(
    clean_root: Path,
    env_root: Path,
    selected: dict[str, list[str]],
    output_path: Path,
) -> None:
    splits = ["train", "val", "test"]
    rows = len(splits)
    cols = max(len(selected.get(split, [])) for split in splits)
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.6 * rows), squeeze=False)

    for r, split in enumerate(splits):
        for c in range(cols):
            ax = axes[r, c]
            ax.axis("off")
            if c >= len(selected.get(split, [])):
                continue
            sample_id = selected[split][c]
            row = load_sample_row(clean_root, split, sample_id)
            payload = load_vector_payload(env_root, split, sample_id)
            hist_x = json_array(row["hist_x_json"])
            hist_y = json_array(row["hist_y_json"])
            fut_x = json_array(row["fut_x_json"])
            fut_y = json_array(row["fut_y_json"])
            for item in payload["shoreline"]:
                xy = np.asarray(item["xy"], dtype=np.float32)
                ax.plot(xy[:, 0], xy[:, 1], color="#1f77b4", linewidth=0.8, alpha=0.9)
            for item in payload["waterfront"]:
                xy = np.asarray(item["xy"], dtype=np.float32)
                ax.plot(xy[:, 0], xy[:, 1], color="#ff7f0e", linewidth=0.8, alpha=0.9)
            ax.plot(hist_x, hist_y, color="#2ca02c", linewidth=1.0)
            ax.plot(fut_x, fut_y, color="#d62728", linewidth=1.0)
            ax.scatter([0.0], [0.0], color="black", s=10)
            ax.set_title(f"{split} #{c + 1}", fontsize=10)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.18)
            ax.tick_params(labelsize=7)

    fig.suptitle("Environment Gallery: Multiple Samples Across Splits", fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_summary(args.env_root, args.output_dir / "environment_summary.png")
    plot_anchor_map(args.env_root, args.output_dir / "anchor_map.png")
    selected = select_example_ids(args.env_root)
    gallery = select_gallery_examples(args.env_root, args.gallery_per_split)
    plot_gallery(args.clean_root, args.env_root, gallery, args.output_dir / "environment_gallery.png")
    for split, sample_id in selected.items():
        plot_example(
            clean_root=args.clean_root,
            env_root=args.env_root,
            split=split,
            sample_id=sample_id,
            output_path=args.output_dir / f"{split}_example_{sample_id}.png",
        )
    print(json.dumps({"selected_examples": selected, "gallery_examples": gallery}, indent=2))


if __name__ == "__main__":
    main()
