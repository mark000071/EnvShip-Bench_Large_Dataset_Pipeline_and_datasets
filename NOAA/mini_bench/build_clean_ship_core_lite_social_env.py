#!/usr/bin/env python3
"""Build a combined social + environment package for clean_ship_core_lite_v1.

This builder keeps the exact clean split membership and annotates each sample with:

- recovered absolute anchor position at `hist_end_ts`
- numeric environment features from `environment_v1`
- nearby ship interaction context recovered from stage10 trajectories

The output is intended to support:

- trajectory-only forecasting
- environment-aware forecasting
- multi-ship social interaction forecasting
- joint social + environment modeling
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CLEAN_ROOT = Path(
    "/mnt/nfs/kun/DeepJSCC/NOAA_ship_trajectory_datasets/mini_bench/clean_ship_core_lite_v1"
)
DEFAULT_STAGE10_ROOT = Path(
    "/mnt/nfs/kun/DeepJSCC/NOAA_ship_trajectory_datasets/data_interim/10_minlen_filtered/partitions"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-root", type=Path, default=DEFAULT_CLEAN_ROOT)
    parser.add_argument("--environment-root", type=Path, default=None)
    parser.add_argument("--stage10-root", type=Path, default=DEFAULT_STAGE10_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--radius-m", type=float, default=3000.0)
    parser.add_argument("--max-neighbors", type=int, default=8)
    parser.add_argument("--bucket-count", type=int, default=64)
    parser.add_argument("--min-neighbors-flag", type=int, default=2)
    parser.add_argument("--skip-snapshot-build", action="store_true")
    parser.add_argument("--log-every-buckets", type=int, default=8)
    return parser.parse_args()


def normalize_ts(value) -> str:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%dT%H:%M:%S+00:00")


def bucket_id(ts: str, bucket_count: int) -> int:
    return hash(ts) % bucket_count


def meters_per_deg_lat() -> float:
    return 110540.0


def meters_per_deg_lon(lat_deg: float) -> float:
    return 111320.0 * math.cos(math.radians(lat_deg))


def latlon_to_local_xy(lat_series, lon_series, lat0: float, lon0: float):
    mx = meters_per_deg_lon(lat0)
    my = meters_per_deg_lat()
    xs = [(lon - lon0) * mx for lon in lon_series]
    ys = [(lat - lat0) * my for lat in lat_series]
    return xs, ys


def sog_cog_to_velocity(sog_knots: float, cog_deg: float) -> tuple[float, float]:
    if pd.isna(sog_knots) or pd.isna(cog_deg):
        return 0.0, 0.0
    speed = float(sog_knots) * 0.514444
    rad = math.radians(float(cog_deg))
    vx = speed * math.sin(rad)
    vy = speed * math.cos(rad)
    return vx, vy


def compute_cpa_tcpa(rel_x: float, rel_y: float, rel_vx: float, rel_vy: float) -> tuple[float | None, float | None]:
    v2 = rel_vx * rel_vx + rel_vy * rel_vy
    if v2 <= 1e-8:
        return None, None
    tcpa = -((rel_x * rel_vx) + (rel_y * rel_vy)) / v2
    cpa_x = rel_x + tcpa * rel_vx
    cpa_y = rel_y + tcpa * rel_vy
    cpa = math.sqrt(cpa_x * cpa_x + cpa_y * cpa_y)
    return cpa, tcpa


def density_bin(neighbor_count: int) -> str:
    if neighbor_count >= 7:
        return "dense"
    if neighbor_count >= 4:
        return "medium"
    if neighbor_count >= 1:
        return "sparse"
    return "isolated"


def load_clean_split(path: Path, split: str) -> pd.DataFrame:
    frame = pd.read_csv(path, compression="gzip", low_memory=False)
    frame["split"] = split
    frame["hist_end_ts"] = frame["hist_end_ts"].map(normalize_ts)
    frame["pred_end_ts"] = frame["pred_end_ts"].map(normalize_ts)
    return frame


def load_clean_targets(clean_root: Path) -> tuple[pd.DataFrame, dict[int, list[dict]], set[str], dict[str, int]]:
    frames = []
    for split in ("train", "val", "test"):
        frames.append(load_clean_split(clean_root / split / "part-000.csv.gz", split))
    all_targets = pd.concat(frames, ignore_index=True)
    split_counts = all_targets.groupby("split").size().to_dict()
    targets_by_bucket: dict[int, list[dict]] = defaultdict(list)
    target_ts_set: set[str] = set()
    for row in all_targets.to_dict(orient="records"):
        ts = row["hist_end_ts"]
        target_ts_set.add(ts)
        targets_by_bucket[bucket_id(ts, 64)].append(row)
    return all_targets, targets_by_bucket, target_ts_set, split_counts


class BucketCsvWriters:
    def __init__(self, base_dir: Path, fieldnames: list[str]):
        self.base_dir = base_dir
        self.handles: dict[int, gzip.GzipFile] = {}
        self.writers: dict[int, csv.DictWriter] = {}
        self.fieldnames = fieldnames

    def write(self, bucket: int, row: dict) -> None:
        if bucket not in self.writers:
            path = self.base_dir / f"snapshots-bucket-{bucket:03d}.csv.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            exists = path.exists()
            handle = gzip.open(path, "at", newline="", encoding="utf-8")
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            if not exists:
                writer.writeheader()
            self.handles[bucket] = handle
            self.writers[bucket] = writer
        self.writers[bucket].writerow(row)

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()


def snapshot_from_segment(seg: pd.DataFrame, idx: int) -> dict:
    hist = seg.iloc[idx - 29 : idx + 1]
    row = seg.iloc[idx]
    return {
        "timestamp_utc": normalize_ts(row["timestamp_utc"]),
        "mmsi": int(row["mmsi"]),
        "segment_id": str(row["segment_id"]),
        "ship_type": "" if pd.isna(row.get("ship_type")) else str(row["ship_type"]),
        "lat": float(row["lat"]),
        "lon": float(row["lon"]),
        "sog": None if pd.isna(row.get("sog")) else float(row["sog"]),
        "cog": None if pd.isna(row.get("cog")) else float(row["cog"]),
        "heading": None if pd.isna(row.get("heading")) else float(row["heading"]),
        "hist_lat_json": json.dumps(hist["lat"].astype(float).tolist()),
        "hist_lon_json": json.dumps(hist["lon"].astype(float).tolist()),
        "hist_sog_json": json.dumps([None if pd.isna(v) else float(v) for v in hist.get("sog", pd.Series([None] * len(hist))).tolist()]),
        "hist_cog_json": json.dumps([None if pd.isna(v) else float(v) for v in hist.get("cog", pd.Series([None] * len(hist))).tolist()]),
        "hist_heading_json": json.dumps([None if pd.isna(v) else float(v) for v in hist.get("heading", pd.Series([None] * len(hist))).tolist()]),
    }


def build_snapshot_buckets(stage10_root: Path, target_ts_set: set[str], bucket_count: int, snapshot_dir: Path) -> None:
    fieldnames = [
        "timestamp_utc",
        "mmsi",
        "segment_id",
        "ship_type",
        "lat",
        "lon",
        "sog",
        "cog",
        "heading",
        "hist_lat_json",
        "hist_lon_json",
        "hist_sog_json",
        "hist_cog_json",
        "hist_heading_json",
    ]
    writers = BucketCsvWriters(snapshot_dir, fieldnames)
    for part_path in sorted(stage10_root.glob("part-*.csv.gz")):
        print(f"[snapshots] scanning {part_path.name}")
        frame = pd.read_csv(part_path, compression="gzip", low_memory=False)
        frame["timestamp_utc"] = frame["timestamp_utc"].map(normalize_ts)
        frame = frame.sort_values(["segment_id", "timestamp_utc"]).reset_index(drop=True)
        for _, seg in frame.groupby("segment_id", sort=False):
            if len(seg) < 30:
                continue
            ts_values = seg["timestamp_utc"].tolist()
            for idx in range(29, len(seg)):
                ts = ts_values[idx]
                if ts not in target_ts_set:
                    continue
                writers.write(bucket_id(ts, bucket_count), snapshot_from_segment(seg, idx))
    writers.close()


def empty_social_payload(target: dict) -> dict:
    out = dict(target)
    out["target_global_lat"] = None
    out["target_global_lon"] = None
    out["target_sog_hist_end"] = None
    out["target_cog_hist_end"] = None
    out["target_heading_hist_end"] = None
    out["neighbor_count_total"] = 0
    out["neighbor_count_used"] = 0
    out["neighbor_density_bin"] = "isolated"
    out["neighbor_mmsi_json"] = "[]"
    out["neighbor_segment_id_json"] = "[]"
    out["neighbor_ship_type_json"] = "[]"
    out["neighbor_distance_m_json"] = "[]"
    out["neighbor_rel_hist_x_json"] = "[]"
    out["neighbor_rel_hist_y_json"] = "[]"
    out["neighbor_hist_sog_json"] = "[]"
    out["neighbor_hist_cog_json"] = "[]"
    out["neighbor_hist_heading_json"] = "[]"
    out["neighbor_rel_vx_mps_json"] = "[]"
    out["neighbor_rel_vy_mps_json"] = "[]"
    out["neighbor_cpa_m_json"] = "[]"
    out["neighbor_tcpa_s_json"] = "[]"
    out["social_context_available"] = 0
    out["interaction_candidate"] = 0
    out["min_neighbor_distance_m"] = None
    out["mean_neighbor_distance_m"] = None
    out["min_cpa_m"] = None
    out["min_abs_tcpa_s"] = None
    return out


def enrich_target(target: dict, target_snap: dict, neighbor_snaps: list[dict], radius_m: float, max_neighbors: int, min_neighbors_flag: int) -> dict:
    lat0 = float(target_snap["lat"])
    lon0 = float(target_snap["lon"])
    target_vx, target_vy = sog_cog_to_velocity(target_snap["sog"], target_snap["cog"])

    distances = []
    for snap in neighbor_snaps:
        dx = (float(snap["lon"]) - lon0) * meters_per_deg_lon(lat0)
        dy = (float(snap["lat"]) - lat0) * meters_per_deg_lat()
        dist = math.sqrt(dx * dx + dy * dy)
        if dist <= radius_m and int(snap["mmsi"]) != int(target["mmsi"]):
            distances.append((dist, dx, dy, snap))

    distances.sort(key=lambda item: item[0])
    chosen = distances[:max_neighbors]
    out = empty_social_payload(target)
    out["target_global_lat"] = lat0
    out["target_global_lon"] = lon0
    out["target_sog_hist_end"] = target_snap["sog"]
    out["target_cog_hist_end"] = target_snap["cog"]
    out["target_heading_hist_end"] = target_snap["heading"]
    out["neighbor_count_total"] = len(distances)
    out["neighbor_count_used"] = len(chosen)
    out["neighbor_density_bin"] = density_bin(len(distances))
    out["social_context_available"] = 1
    out["interaction_candidate"] = int(len(distances) >= min_neighbors_flag)

    neighbor_mmsi = []
    neighbor_segment_ids = []
    neighbor_ship_types = []
    neighbor_distance_m = []
    neighbor_rel_hist_x = []
    neighbor_rel_hist_y = []
    neighbor_hist_sog = []
    neighbor_hist_cog = []
    neighbor_hist_heading = []
    neighbor_rel_vx = []
    neighbor_rel_vy = []
    neighbor_cpa = []
    neighbor_tcpa = []

    for dist, _dx, _dy, snap in chosen:
        hist_lat = json.loads(snap["hist_lat_json"])
        hist_lon = json.loads(snap["hist_lon_json"])
        rel_x, rel_y = latlon_to_local_xy(hist_lat, hist_lon, lat0, lon0)
        vx, vy = sog_cog_to_velocity(snap["sog"], snap["cog"])
        rel_vx = vx - target_vx
        rel_vy = vy - target_vy
        cpa, tcpa = compute_cpa_tcpa(rel_x[-1], rel_y[-1], rel_vx, rel_vy)

        neighbor_mmsi.append(int(snap["mmsi"]))
        neighbor_segment_ids.append(str(snap["segment_id"]))
        neighbor_ship_types.append(str(snap["ship_type"]))
        neighbor_distance_m.append(float(dist))
        neighbor_rel_hist_x.append(rel_x)
        neighbor_rel_hist_y.append(rel_y)
        neighbor_hist_sog.append(json.loads(snap["hist_sog_json"]))
        neighbor_hist_cog.append(json.loads(snap["hist_cog_json"]))
        neighbor_hist_heading.append(json.loads(snap["hist_heading_json"]))
        neighbor_rel_vx.append(rel_vx)
        neighbor_rel_vy.append(rel_vy)
        neighbor_cpa.append(cpa)
        neighbor_tcpa.append(tcpa)

    out["neighbor_mmsi_json"] = json.dumps(neighbor_mmsi)
    out["neighbor_segment_id_json"] = json.dumps(neighbor_segment_ids)
    out["neighbor_ship_type_json"] = json.dumps(neighbor_ship_types)
    out["neighbor_distance_m_json"] = json.dumps(neighbor_distance_m)
    out["neighbor_rel_hist_x_json"] = json.dumps(neighbor_rel_hist_x)
    out["neighbor_rel_hist_y_json"] = json.dumps(neighbor_rel_hist_y)
    out["neighbor_hist_sog_json"] = json.dumps(neighbor_hist_sog)
    out["neighbor_hist_cog_json"] = json.dumps(neighbor_hist_cog)
    out["neighbor_hist_heading_json"] = json.dumps(neighbor_hist_heading)
    out["neighbor_rel_vx_mps_json"] = json.dumps(neighbor_rel_vx)
    out["neighbor_rel_vy_mps_json"] = json.dumps(neighbor_rel_vy)
    out["neighbor_cpa_m_json"] = json.dumps(neighbor_cpa)
    out["neighbor_tcpa_s_json"] = json.dumps(neighbor_tcpa)
    out["min_neighbor_distance_m"] = min(neighbor_distance_m) if neighbor_distance_m else None
    out["mean_neighbor_distance_m"] = float(np.mean(neighbor_distance_m)) if neighbor_distance_m else None
    valid_cpa = [v for v in neighbor_cpa if v is not None]
    valid_tcpa = [abs(v) for v in neighbor_tcpa if v is not None]
    out["min_cpa_m"] = min(valid_cpa) if valid_cpa else None
    out["min_abs_tcpa_s"] = min(valid_tcpa) if valid_tcpa else None
    return out


def process_buckets(
    targets_by_bucket: dict[int, list[dict]],
    snapshot_dir: Path,
    bucket_count: int,
    radius_m: float,
    max_neighbors: int,
    min_neighbors_flag: int,
    log_every_buckets: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    all_rows = []
    summary_by_split = {
        "train": {"neighbor_counts": [], "density": Counter(), "interaction_candidates": 0, "available_context": 0},
        "val": {"neighbor_counts": [], "density": Counter(), "interaction_candidates": 0, "available_context": 0},
        "test": {"neighbor_counts": [], "density": Counter(), "interaction_candidates": 0, "available_context": 0},
    }

    for bucket in range(bucket_count):
        if bucket and bucket % log_every_buckets == 0:
            print(f"[social] processed buckets={bucket}/{bucket_count}")
        targets = targets_by_bucket.get(bucket, [])
        if not targets:
            continue
        snap_path = snapshot_dir / f"snapshots-bucket-{bucket:03d}.csv.gz"
        grouped_snaps: dict[str, pd.DataFrame] = {}
        if snap_path.exists():
            snaps = pd.read_csv(snap_path, compression="gzip", low_memory=False)
            grouped_snaps = {ts: grp.reset_index(drop=True) for ts, grp in snaps.groupby("timestamp_utc", sort=False)}

        for target in targets:
            ts = target["hist_end_ts"]
            grp = grouped_snaps.get(ts)
            if grp is None:
                row = empty_social_payload(target)
            else:
                records = grp.to_dict(orient="records")
                record_index = {(int(rec["mmsi"]), str(rec["segment_id"])): idx for idx, rec in enumerate(records)}
                target_idx = record_index.get((int(target["mmsi"]), str(target["segment_id"])))
                if target_idx is None:
                    row = empty_social_payload(target)
                else:
                    target_snap = records[target_idx]
                    lats = grp["lat"].astype(float).to_numpy()
                    lons = grp["lon"].astype(float).to_numpy()
                    mmsis = grp["mmsi"].astype(int).to_numpy()
                    lat0 = float(lats[target_idx])
                    lon0 = float(lons[target_idx])
                    dx = (lons - lon0) * meters_per_deg_lon(lat0)
                    dy = (lats - lat0) * meters_per_deg_lat()
                    dist = np.sqrt(dx * dx + dy * dy)
                    mask = (dist <= radius_m) & (mmsis != int(target["mmsi"]))
                    neighbor_indices = np.where(mask)[0]
                    ordered = neighbor_indices[np.argsort(dist[neighbor_indices])] if len(neighbor_indices) else []
                    neighbor_records = [records[int(i)] for i in ordered]
                    row = enrich_target(target, target_snap, neighbor_records, radius_m, max_neighbors, min_neighbors_flag)

            split = row["split"]
            summary_by_split[split]["neighbor_counts"].append(int(row["neighbor_count_total"]))
            summary_by_split[split]["density"][row["neighbor_density_bin"]] += 1
            summary_by_split[split]["interaction_candidates"] += int(row["interaction_candidate"])
            summary_by_split[split]["available_context"] += int(row["social_context_available"])
            all_rows.append(row)

    summary = {}
    for split, stats in summary_by_split.items():
        neighbor_counts = stats.pop("neighbor_counts")
        summary[split] = {
            "num_samples": int(len(neighbor_counts)),
            "social_context_available": int(stats["available_context"]),
            "interaction_candidate_count": int(stats["interaction_candidates"]),
            "density_bin_counts": dict(stats["density"]),
            "mean_neighbor_count": float(np.mean(neighbor_counts)) if neighbor_counts else 0.0,
            "samples_with_neighbors": int(sum(count > 0 for count in neighbor_counts)),
        }
    return pd.DataFrame(all_rows), summary


def load_environment_features(environment_root: Path) -> pd.DataFrame:
    path = environment_root / "all_environment_features.csv"
    if path.exists():
        frame = pd.read_csv(path)
    else:
        frames = []
        for split in ("train", "val", "test"):
            split_path = environment_root / "features" / split / "environment_features.csv"
            split_frame = pd.read_csv(split_path)
            split_frame["split"] = split
            frames.append(split_frame)
        frame = pd.concat(frames, ignore_index=True)
    return frame


def write_readme(output_root: Path, args: argparse.Namespace) -> None:
    text = f"""# Social Environment Package v1

This package augments `clean_ship_core_lite_v1` with both:

- sample-centric environment context from `environment_v1`
- nearby-ship social interaction context recovered from stage10 AIS trajectories

Key properties:

- keeps the exact clean split membership
- preserves all clean samples, including isolated cases
- marks interaction-rich samples with `interaction_candidate`

Social config:

- radius: {args.radius_m} m
- max neighbors exported: {args.max_neighbors}
- interaction flag threshold: {args.min_neighbors_flag}
- snapshot buckets: {args.bucket_count}
"""
    (output_root / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    environment_root = args.environment_root or (args.clean_root / "environment_v1")
    output_root = args.output_root or (args.clean_root / "social_env_v1")
    output_root.mkdir(parents=True, exist_ok=True)
    snapshot_dir = output_root / "_snapshot_buckets"

    print("[build] loading clean targets")
    all_targets, targets_by_bucket, target_ts_set, split_counts = load_clean_targets(args.clean_root)

    remapped_targets_by_bucket: dict[int, list[dict]] = defaultdict(list)
    for row in all_targets.to_dict(orient="records"):
        remapped_targets_by_bucket[bucket_id(row["hist_end_ts"], args.bucket_count)].append(row)

    if not args.skip_snapshot_build:
        print("[build] building snapshot buckets")
        build_snapshot_buckets(args.stage10_root, target_ts_set, args.bucket_count, snapshot_dir)

    print("[build] computing social context")
    social_df, social_summary = process_buckets(
        targets_by_bucket=remapped_targets_by_bucket,
        snapshot_dir=snapshot_dir,
        bucket_count=args.bucket_count,
        radius_m=args.radius_m,
        max_neighbors=args.max_neighbors,
        min_neighbors_flag=args.min_neighbors_flag,
        log_every_buckets=args.log_every_buckets,
    )
    social_df.to_csv(output_root / "all_social_features.csv", index=False)

    print("[build] loading environment features")
    env_df = load_environment_features(environment_root)
    keep_env_cols = [col for col in env_df.columns if col not in {"segment_id", "hist_end_ts"}]
    env_df = env_df[keep_env_cols].copy()

    merged_all = social_df.merge(env_df, on=["sample_id", "split"], how="left", validate="one_to_one")
    missing_env = int(merged_all["anchor_lat"].isna().sum()) if "anchor_lat" in merged_all.columns else len(merged_all)
    if missing_env:
        raise RuntimeError(f"Missing environment rows for {missing_env} samples")

    merged_all.to_csv(output_root / "all_social_env_features.csv", index=False)

    print("[build] exporting per-split outputs")
    split_summaries = {}
    for split in ("train", "val", "test"):
        source_path = args.clean_root / split / "part-000.csv.gz"
        clean_df = pd.read_csv(source_path, compression="gzip", low_memory=False)
        split_features = merged_all.loc[merged_all["split"] == split].copy()
        (output_root / "features" / split).mkdir(parents=True, exist_ok=True)
        split_features.to_csv(output_root / "features" / split / "social_env_features.csv", index=False)
        sample_ids = split_features["sample_id"].astype(str)
        (output_root / "sample_ids").mkdir(parents=True, exist_ok=True)
        sample_ids.to_csv(output_root / "sample_ids" / f"{split}_sample_ids.txt", index=False, header=False)
        extra_cols = [col for col in split_features.columns if col not in clean_df.columns]
        augmented = clean_df.merge(split_features[["sample_id", *extra_cols]], on="sample_id", how="left", validate="one_to_one")
        out_path = output_root / "augmented_splits" / split / "part-000.csv.gz"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(out_path, "wt", encoding="utf-8", newline="") as handle:
            augmented.to_csv(handle, index=False)

        split_summary = dict(social_summary[split])
        split_summary["augmented_rows"] = int(len(augmented))
        split_summary["unique_mmsi"] = int(clean_df["mmsi"].nunique()) if "mmsi" in clean_df.columns else 0
        split_summary["unique_segments"] = int(clean_df["segment_id"].astype(str).nunique()) if "segment_id" in clean_df.columns else 0
        split_summaries[split] = split_summary

    summary = {
        "config": {
            "clean_root": str(args.clean_root),
            "environment_root": str(environment_root),
            "stage10_root": str(args.stage10_root),
            "output_root": str(output_root),
            "radius_m": args.radius_m,
            "max_neighbors": args.max_neighbors,
            "min_neighbors_flag": args.min_neighbors_flag,
            "bucket_count": args.bucket_count,
        },
        "source_target_counts": split_counts,
        **split_summaries,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_readme(output_root, args)
    print("[build] finished")


if __name__ == "__main__":
    main()
