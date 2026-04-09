#!/usr/bin/env python3
"""Preview benchmark samples as CSV and trajectory figures."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.pipeline_utils import latlon_to_local_xy


@dataclass
class SampleRecord:
    row: dict
    hist_x: np.ndarray
    hist_y: np.ndarray
    fut_x: np.ndarray
    fut_y: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-file", type=Path, required=True)
    parser.add_argument("--sample-id", type=str, default=None)
    parser.add_argument("--stage10-dir", type=Path, default=Path("tmp/pilot_2025_09_01/10"))
    parser.add_argument("--output-dir", type=Path, default=Path("visualization/output"))
    parser.add_argument("--nearby-radius-m", type=float, default=2000.0)
    parser.add_argument("--history-points", type=int, default=15)
    return parser.parse_args()


def _json_array(value: str) -> np.ndarray:
    return np.asarray(json.loads(value), dtype=float)


def load_sample(sample_file: Path, sample_id: str | None) -> SampleRecord:
    with gzip.open(sample_file, "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        first_row = None
        for row in reader:
            if first_row is None:
                first_row = row
            if sample_id is None or row["sample_id"] == sample_id:
                return SampleRecord(
                    row=row,
                    hist_x=_json_array(row["hist_x_json"]),
                    hist_y=_json_array(row["hist_y_json"]),
                    fut_x=_json_array(row["fut_x_json"]),
                    fut_y=_json_array(row["fut_y_json"]),
                )
    if first_row is None:
        raise ValueError(f"No rows found in {sample_file}")
    raise ValueError(f"sample_id {sample_id!r} not found in {sample_file}")


def write_preview_csv(sample: SampleRecord, out_path: Path) -> None:
    hist_len = len(sample.hist_x)
    fut_len = len(sample.fut_x)
    hist_interp = json.loads(sample.row["hist_interp_json"])
    fut_interp = json.loads(sample.row["fut_interp_json"])
    hist_grid = json.loads(sample.row["hist_grid_interp_json"])
    fut_grid = json.loads(sample.row["fut_grid_interp_json"])

    rows = []
    for idx in range(hist_len):
        rows.append(
            {
                "phase": "history",
                "step": idx,
                "x_m": float(sample.hist_x[idx]),
                "y_m": float(sample.hist_y[idx]),
                "gap_imputed": bool(hist_interp[idx]),
                "grid_interpolated": bool(hist_grid[idx]),
            }
        )
    for idx in range(fut_len):
        rows.append(
            {
                "phase": "future",
                "step": idx,
                "x_m": float(sample.fut_x[idx]),
                "y_m": float(sample.fut_y[idx]),
                "gap_imputed": bool(fut_interp[idx]),
                "grid_interpolated": bool(fut_grid[idx]),
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def plot_single_sample(sample: SampleRecord, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(sample.hist_x, sample.hist_y, "-o", color="#1f77b4", markersize=3, label="Observed 10 min")
    ax.plot(sample.fut_x, sample.fut_y, "-o", color="#d62728", markersize=3, label="Future 10 min")
    ax.scatter([sample.hist_x[-1]], [sample.hist_y[-1]], color="black", s=35, zorder=5, label="Prediction origin")
    ax.set_title(f"Sample {sample.row['sample_id']}\n{sample.row['ship_type']} / {sample.row['quality_tier']}")
    ax.set_xlabel("x (meters)")
    ax.set_ylabel("y (meters)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def load_target_segment(stage10_dir: Path, segment_id: str) -> pd.DataFrame:
    for path in sorted((stage10_dir / "partitions").glob("part-*.csv.gz")):
        df = pd.read_csv(path, compression="gzip", low_memory=False)
        target = df.loc[df["segment_id"] == segment_id].copy()
        if not target.empty:
            target["timestamp_utc"] = pd.to_datetime(target["timestamp_utc"], utc=True)
            return target.sort_values("timestamp_utc").reset_index(drop=True)
    raise FileNotFoundError(f"segment_id {segment_id} not found under {stage10_dir}")


def find_nearby_ships(
    stage10_dir: Path,
    target_segment: pd.DataFrame,
    hist_end_ts: pd.Timestamp,
    radius_m: float,
    history_points: int,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    target_row = target_segment.loc[target_segment["timestamp_utc"] == hist_end_ts]
    if target_row.empty:
        idx = int((target_segment["timestamp_utc"] - hist_end_ts).abs().argmin())
        target_row = target_segment.iloc[[idx]]
    ref_lat = float(target_row["lat"].iloc[0])
    ref_lon = float(target_row["lon"].iloc[0])

    target_history = target_segment.loc[target_segment["timestamp_utc"] <= hist_end_ts].tail(history_points).copy()
    target_x, target_y = latlon_to_local_xy(
        target_history["lat"].to_numpy(dtype=float),
        target_history["lon"].to_numpy(dtype=float),
        ref_lat,
        ref_lon,
    )
    target_history["x_m"] = target_x
    target_history["y_m"] = target_y
    nearby_tracks: list[pd.DataFrame] = []
    for path in sorted((stage10_dir / "partitions").glob("part-*.csv.gz")):
        df = pd.read_csv(path, compression="gzip", low_memory=False)
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        snap = df.loc[df["timestamp_utc"] == hist_end_ts].copy()
        if snap.empty:
            continue
        x_snap, y_snap = latlon_to_local_xy(
            snap["lat"].to_numpy(dtype=float),
            snap["lon"].to_numpy(dtype=float),
            ref_lat,
            ref_lon,
        )
        snap["x_m"] = x_snap
        snap["y_m"] = y_snap
        snap["dist_m"] = np.hypot(snap["x_m"], snap["y_m"])
        keep = snap.loc[(snap["dist_m"] <= radius_m) & (snap["segment_id"] != target_segment["segment_id"].iloc[0])]
        for segment_id in keep["segment_id"].astype(str).tolist():
            seg = df.loc[df["segment_id"] == segment_id].copy()
            seg = seg.loc[seg["timestamp_utc"] <= hist_end_ts].tail(history_points).copy()
            if seg.empty:
                continue
            x, y = latlon_to_local_xy(
                seg["lat"].to_numpy(dtype=float),
                seg["lon"].to_numpy(dtype=float),
                ref_lat,
                ref_lon,
            )
            seg["x_m"] = x
            seg["y_m"] = y
            nearby_tracks.append(seg)
    dedup = {}
    for seg in nearby_tracks:
        dedup[str(seg["segment_id"].iloc[0])] = seg
    return target_history, list(dedup.values())


def plot_interaction_snapshot(
    sample: SampleRecord,
    target_history: pd.DataFrame,
    nearby_tracks: list[pd.DataFrame],
    out_path: Path,
    radius_m: float,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    for seg in nearby_tracks:
        ax.plot(seg["x_m"], seg["y_m"], color="#7f7f7f", alpha=0.55, linewidth=1.2)
        ax.scatter(seg["x_m"].iloc[-1], seg["y_m"].iloc[-1], color="#7f7f7f", s=12)

    if not target_history.empty:
        ax.plot(target_history["x_m"], target_history["y_m"], "-o", color="#1f77b4", markersize=3, label="Target observed context")
    ax.plot(sample.fut_x, sample.fut_y, "-o", color="#d62728", markersize=3, label="Target future target")
    ax.scatter([0.0], [0.0], color="black", s=40, zorder=6, label="Target at hist_end")

    ax.set_title(
        f"Interaction Snapshot around {sample.row['hist_end_ts']}\n"
        f"Nearby ships within {int(radius_m)} m: {len(nearby_tracks)}"
    )
    ax.set_xlabel("x (meters, target-centered)")
    ax.set_ylabel("y (meters, target-centered)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_metadata(sample: SampleRecord, nearby_tracks: list[pd.DataFrame], out_path: Path) -> None:
    payload = {
        "sample_id": sample.row["sample_id"],
        "mmsi": int(sample.row["mmsi"]),
        "segment_id": sample.row["segment_id"],
        "ship_type": sample.row["ship_type"],
        "ship_class": sample.row["ship_class"],
        "quality_tier": sample.row["quality_tier"],
        "hist_end_ts": sample.row["hist_end_ts"],
        "pred_end_ts": sample.row["pred_end_ts"],
        "interp_ratio_total": float(sample.row["interp_ratio_total"]),
        "grid_interp_ratio_total": float(sample.row["grid_interp_ratio_total"]),
        "hist_displacement_m": float(sample.row["hist_displacement_m"]),
        "fut_displacement_m": float(sample.row["fut_displacement_m"]),
        "nearby_ship_count_at_hist_end": len(nearby_tracks),
        "nearby_segment_ids": [str(seg["segment_id"].iloc[0]) for seg in nearby_tracks],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample = load_sample(args.sample_file, args.sample_id)

    preview_csv = args.output_dir / "sample_preview.csv"
    plot_png = args.output_dir / "sample_trajectory.png"
    interaction_png = args.output_dir / "interaction_snapshot.png"
    metadata_json = args.output_dir / "sample_metadata.json"

    write_preview_csv(sample, preview_csv)
    plot_single_sample(sample, plot_png)

    target_segment = load_target_segment(args.stage10_dir, sample.row["segment_id"])
    hist_end_ts = pd.to_datetime(sample.row["hist_end_ts"], utc=True)
    target_history, nearby_tracks = find_nearby_ships(
        args.stage10_dir,
        target_segment,
        hist_end_ts,
        args.nearby_radius_m,
        args.history_points,
    )
    plot_interaction_snapshot(sample, target_history, nearby_tracks, interaction_png, args.nearby_radius_m)
    write_metadata(sample, nearby_tracks, metadata_json)

    print(preview_csv)
    print(plot_png)
    print(interaction_png)
    print(metadata_json)


if __name__ == "__main__":
    main()
