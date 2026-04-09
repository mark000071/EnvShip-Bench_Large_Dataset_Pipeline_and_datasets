#!/usr/bin/env python3
"""Build a representative Ship-Core-Lite subset from benchmark/core."""

from __future__ import annotations

import argparse
import gzip
import heapq
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


KNOTS_PER_MPS = 1.9438444924406


@dataclass
class Candidate:
    hash_value: int
    sample_id: str
    mmsi: int
    segment_id: str
    sample_step: int
    ship_group: str
    speed_bin: str
    motion_bin: str
    turn_bin: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Ship-Core-Lite mini benchmark.")
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-target", type=int, default=32000)
    parser.add_argument("--val-target", type=int, default=4000)
    parser.add_argument("--test-target", type=int, default=4000)
    parser.add_argument("--chunk-size", type=int, default=20000)
    parser.add_argument("--oversample-factor", type=float, default=6.0)
    parser.add_argument("--max-per-mmsi", type=int, default=25)
    parser.add_argument("--max-per-segment", type=int, default=8)
    parser.add_argument("--min-segment-step-gap", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1666)
    return parser.parse_args()


def normalize_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip().lower()


def ship_group_from_type(ship_type: object, ship_class: object | None = None) -> str:
    text = normalize_text(ship_type)
    text2 = normalize_text(ship_class)
    full = f"{text} {text2}".strip()
    if any(k in full for k in ["cargo", "tanker", "bulk", "container"]):
        return "cargo_tanker"
    if any(k in full for k in ["passenger", "ferry"]):
        return "passenger_ferry"
    if "fishing" in full:
        return "fishing"
    if any(k in full for k in ["tug", "service", "pilot", "tow"]):
        return "tug_service"
    if any(k in full for k in ["sailing", "pleasure", "yacht"]):
        return "sailing_leisure"
    return "other_unknown"


def avg_speed_knots(hist_disp_m: float, fut_disp_m: float) -> float:
    hist_speed = float(hist_disp_m) / 600.0 * KNOTS_PER_MPS
    fut_speed = float(fut_disp_m) / 600.0 * KNOTS_PER_MPS
    return 0.5 * (hist_speed + fut_speed)


def speed_bin_from_knots(speed_knots: float) -> str:
    if speed_knots < 5:
        return "0_5"
    if speed_knots < 10:
        return "5_10"
    if speed_knots < 15:
        return "10_15"
    if speed_knots < 20:
        return "15_20"
    return "gt20"


def motion_bin_from_disp(fut_disp_m: float) -> str:
    if fut_disp_m < 600:
        return "low"
    if fut_disp_m < 1500:
        return "mid"
    return "high"


def stable_hash_series(series: pd.Series) -> np.ndarray:
    return pd.util.hash_pandas_object(series, index=False).to_numpy(dtype=np.uint64)


def parse_sample_step(sample_id: str) -> int:
    try:
        return int(sample_id.rsplit("_", 1)[-1])
    except Exception:
        return -1


def parse_angle_array(sin_json: str, cos_json: str) -> np.ndarray:
    s = np.fromstring(sin_json.strip()[1:-1], sep=",", dtype=np.float64)
    c = np.fromstring(cos_json.strip()[1:-1], sep=",", dtype=np.float64)
    n = min(len(s), len(c))
    if n == 0:
        return np.array([], dtype=np.float64)
    angles = np.degrees(np.arctan2(s[:n], c[:n]))
    return angles


def wrap_angle_diff_deg(a: float, b: float) -> float:
    diff = (b - a + 180.0) % 360.0 - 180.0
    return abs(diff)


def turn_bin_from_hist(sin_json: str, cos_json: str) -> str:
    try:
        angles = parse_angle_array(sin_json, cos_json)
    except Exception:
        return "unknown_turn"
    if len(angles) < 6:
        return "unknown_turn"
    first = float(np.nanmean(angles[:10]))
    last = float(np.nanmean(angles[-10:]))
    if np.isnan(first) or np.isnan(last):
        return "unknown_turn"
    delta = wrap_angle_diff_deg(first, last)
    if delta < 15:
        return "straight"
    if delta < 60:
        return "gentle_turn"
    return "sharp_turn"


def quota_from_counts(counts: dict[tuple, int], target: int) -> dict[tuple, int]:
    if target <= 0 or not counts:
        return {k: 0 for k in counts}
    # Sqrt weighting softens the dominance of very common traffic modes while
    # still keeping the overall subset close to the underlying distribution.
    weights = {k: math.sqrt(v) for k, v in counts.items() if v > 0}
    total = sum(weights.values())
    raw = {k: weights[k] / total * target for k in weights}
    base = {k: min(counts[k], int(math.floor(raw[k]))) for k in weights}
    remaining = target - sum(base.values())
    frac_order = sorted(
        weights.keys(),
        key=lambda k: (raw[k] - math.floor(raw[k]), counts[k] - base[k]),
        reverse=True,
    )
    for key in frac_order:
        if remaining <= 0:
            break
        if base[key] < counts[key]:
            base[key] += 1
            remaining -= 1
    return {k: base.get(k, 0) for k in counts}


def coarse_key(ship_group: str, speed_bin: str, motion_bin: str) -> tuple[str, str, str]:
    return (ship_group, speed_bin, motion_bin)


def select_candidate_pools(
    benchmark_split_dir: Path,
    coarse_quotas: dict[tuple, int],
    chunk_size: int,
    oversample_factor: float,
) -> dict[tuple, list[tuple[int, dict]]]:
    pools: dict[tuple, list[tuple[int, dict]]] = defaultdict(list)
    # Keep a bounded candidate pool per coarse stratum so the builder can scan
    # large benchmark shards without materializing every eligible row in memory.
    capacities = {
        key: max(int(math.ceil(quota * oversample_factor)), quota + 20, 50) if quota > 0 else 0
        for key, quota in coarse_quotas.items()
    }

    files = sorted(benchmark_split_dir.glob("*.csv.gz"))
    for file_path in files:
        for chunk in pd.read_csv(file_path, compression="gzip", chunksize=chunk_size, low_memory=False):
            hashes = stable_hash_series(chunk["sample_id"])
            ship_groups = [
                ship_group_from_type(st, sc)
                for st, sc in zip(chunk["ship_type"], chunk["ship_class"], strict=False)
            ]
            speeds = [
                speed_bin_from_knots(avg_speed_knots(hd, fd))
                for hd, fd in zip(chunk["hist_displacement_m"], chunk["fut_displacement_m"], strict=False)
            ]
            motions = [motion_bin_from_disp(v) for v in chunk["fut_displacement_m"]]
            for i, row in enumerate(chunk.itertuples(index=False)):
                hk = coarse_key(ship_groups[i], speeds[i], motions[i])
                cap = capacities.get(hk, 0)
                if cap <= 0:
                    continue
                meta = {
                    "sample_id": str(row.sample_id),
                    "mmsi": int(row.mmsi),
                    "segment_id": str(row.segment_id),
                    "sample_step": parse_sample_step(str(row.sample_id)),
                    "ship_group": hk[0],
                    "speed_bin": hk[1],
                    "motion_bin": hk[2],
                    "hist_cog_sin_json": row.hist_cog_sin_json,
                    "hist_cog_cos_json": row.hist_cog_cos_json,
                }
                hash_value = int(hashes[i])
                pool = pools[hk]
                if len(pool) < cap:
                    heapq.heappush(pool, (-hash_value, meta))
                else:
                    current_worst = -pool[0][0]
                    if hash_value < current_worst:
                        heapq.heapreplace(pool, (-hash_value, meta))
    for key in list(pools.keys()):
        pools[key] = sorted([(-neg_hash, meta) for neg_hash, meta in pools[key]], key=lambda x: x[0])
    return pools


def finalize_selection(
    candidate_pools: dict[tuple, list[tuple[int, dict]]],
    coarse_quotas: dict[tuple, int],
    max_per_mmsi: int,
    max_per_segment: int,
    min_segment_step_gap: int,
) -> tuple[list[Candidate], dict]:
    selected: list[Candidate] = []
    used_mmsi: Counter[int] = Counter()
    used_segment: Counter[str] = Counter()
    used_steps: defaultdict[str, list[int]] = defaultdict(list)
    split_report: dict[str, dict] = {}

    def can_take(meta: dict) -> bool:
        # These caps keep the compact split from collapsing into a handful of
        # long voyages or a small set of frequently sampled vessels.
        if used_mmsi[meta["mmsi"]] >= max_per_mmsi:
            return False
        if used_segment[meta["segment_id"]] >= max_per_segment:
            return False
        step = meta["sample_step"]
        if step >= 0:
            for prev in used_steps[meta["segment_id"]]:
                if abs(prev - step) < min_segment_step_gap:
                    return False
        return True

    def take(meta: dict, hash_value: int, turn_bin: str) -> None:
        selected.append(
            Candidate(
                hash_value=hash_value,
                sample_id=meta["sample_id"],
                mmsi=meta["mmsi"],
                segment_id=meta["segment_id"],
                sample_step=meta["sample_step"],
                ship_group=meta["ship_group"],
                speed_bin=meta["speed_bin"],
                motion_bin=meta["motion_bin"],
                turn_bin=turn_bin,
            )
        )
        used_mmsi[meta["mmsi"]] += 1
        used_segment[meta["segment_id"]] += 1
        if meta["sample_step"] >= 0:
            used_steps[meta["segment_id"]].append(meta["sample_step"])

    leftovers: list[tuple[int, dict, str]] = []
    for ck, pool in candidate_pools.items():
        quota = coarse_quotas.get(ck, 0)
        enriched: list[tuple[int, dict, str]] = []
        turn_counts: Counter[str] = Counter()
        for hash_value, meta in pool:
            turn_bin = turn_bin_from_hist(meta["hist_cog_sin_json"], meta["hist_cog_cos_json"])
            enriched.append((hash_value, meta, turn_bin))
            turn_counts[turn_bin] += 1
        turn_quotas = quota_from_counts(dict(turn_counts), quota)
        selected_count = 0
        for turn_bin, subquota in sorted(turn_quotas.items()):
            candidates = [item for item in enriched if item[2] == turn_bin]
            taken_here = 0
            for hash_value, meta, tb in candidates:
                if taken_here >= subquota:
                    leftovers.append((hash_value, meta, tb))
                    continue
                if can_take(meta):
                    take(meta, hash_value, tb)
                    taken_here += 1
                    selected_count += 1
                else:
                    leftovers.append((hash_value, meta, tb))
        if selected_count < quota:
            still_need = quota - selected_count
            for hash_value, meta, tb in sorted(leftovers, key=lambda x: x[0]):
                if still_need <= 0:
                    break
                if coarse_key(meta["ship_group"], meta["speed_bin"], meta["motion_bin"]) != ck:
                    continue
                if can_take(meta):
                    take(meta, hash_value, tb)
                    still_need -= 1
            selected_count = quota - still_need
        split_report["|".join(ck)] = {
            "quota": quota,
            "candidate_pool": len(pool),
            "selected": selected_count,
            "turn_counts": dict(turn_counts),
        }

    return selected, split_report


def export_split(
    benchmark_split_dir: Path,
    selected_ids: set[str],
    output_path: Path,
    chunk_size: int,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrote_header = False
    kept = 0
    for file_path in sorted(benchmark_split_dir.glob("*.csv.gz")):
        for chunk in pd.read_csv(file_path, compression="gzip", chunksize=chunk_size, low_memory=False):
            subset = chunk[chunk["sample_id"].isin(selected_ids)]
            if subset.empty:
                continue
            kept += len(subset)
            mode = "wt" if not wrote_header else "at"
            with gzip.open(output_path, mode, encoding="utf-8", newline="") as f:
                subset.to_csv(f, index=False, header=not wrote_header)
            wrote_header = True
    return kept


def build_split(
    split: str,
    target: int,
    benchmark_root: Path,
    output_root: Path,
    chunk_size: int,
    oversample_factor: float,
    max_per_mmsi: int,
    max_per_segment: int,
    min_segment_step_gap: int,
) -> dict:
    split_dir = benchmark_root / split
    coarse_counts: Counter[tuple] = Counter()

    files = sorted(split_dir.glob("*.csv.gz"))
    for file_path in files:
        for chunk in pd.read_csv(file_path, compression="gzip", chunksize=chunk_size, low_memory=False):
            ship_groups = [
                ship_group_from_type(st, sc)
                for st, sc in zip(chunk["ship_type"], chunk["ship_class"], strict=False)
            ]
            speeds = [
                speed_bin_from_knots(avg_speed_knots(hd, fd))
                for hd, fd in zip(chunk["hist_displacement_m"], chunk["fut_displacement_m"], strict=False)
            ]
            motions = [motion_bin_from_disp(v) for v in chunk["fut_displacement_m"]]
            for sg, sb, mb in zip(ship_groups, speeds, motions, strict=False):
                coarse_counts[coarse_key(sg, sb, mb)] += 1

    print(f"[build] {split}: counted {sum(coarse_counts.values())} rows across {len(coarse_counts)} coarse strata")
    coarse_quotas = quota_from_counts(dict(coarse_counts), target)
    candidate_pools = select_candidate_pools(split_dir, coarse_quotas, chunk_size, oversample_factor)
    print(f"[build] {split}: built candidate pools for {len(candidate_pools)} strata")
    selected, strata_report = finalize_selection(
        candidate_pools,
        coarse_quotas,
        max_per_mmsi,
        max_per_segment,
        min_segment_step_gap,
    )
    print(f"[build] {split}: selected {len(selected)} candidates before global fill")

    # Global fill if the conservative caps left a small shortfall.
    if len(selected) < target:
        chosen_ids = {item.sample_id for item in selected}
        pool_all = []
        for pool in candidate_pools.values():
            for hash_value, meta in pool:
                if meta["sample_id"] not in chosen_ids:
                    tb = turn_bin_from_hist(meta["hist_cog_sin_json"], meta["hist_cog_cos_json"])
                    pool_all.append((hash_value, meta, tb))
        used_mmsi = Counter(item.mmsi for item in selected)
        used_segment = Counter(item.segment_id for item in selected)
        used_steps: defaultdict[str, list[int]] = defaultdict(list)
        for item in selected:
            if item.sample_step >= 0:
                used_steps[item.segment_id].append(item.sample_step)
        for hash_value, meta, tb in sorted(pool_all, key=lambda x: x[0]):
            if len(selected) >= target:
                break
            if used_mmsi[meta["mmsi"]] >= max_per_mmsi:
                continue
            if used_segment[meta["segment_id"]] >= max_per_segment:
                continue
            if meta["sample_step"] >= 0 and any(
                abs(prev - meta["sample_step"]) < min_segment_step_gap for prev in used_steps[meta["segment_id"]]
            ):
                continue
            selected.append(
                Candidate(
                    hash_value=hash_value,
                    sample_id=meta["sample_id"],
                    mmsi=meta["mmsi"],
                    segment_id=meta["segment_id"],
                    sample_step=meta["sample_step"],
                    ship_group=meta["ship_group"],
                    speed_bin=meta["speed_bin"],
                    motion_bin=meta["motion_bin"],
                    turn_bin=tb,
                )
            )
            used_mmsi[meta["mmsi"]] += 1
            used_segment[meta["segment_id"]] += 1
            if meta["sample_step"] >= 0:
                used_steps[meta["segment_id"]].append(meta["sample_step"])

    selected = sorted(selected, key=lambda x: x.hash_value)[:target]
    selected_ids = {item.sample_id for item in selected}

    split_out_dir = output_root / split
    sample_id_dir = output_root / "sample_ids"
    report_dir = output_root / "reports"
    split_out_dir.mkdir(parents=True, exist_ok=True)
    sample_id_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    exported_rows = export_split(split_dir, selected_ids, split_out_dir / "part-000.csv.gz", chunk_size)
    print(f"[build] {split}: exported {exported_rows} rows")
    with open(sample_id_dir / f"{split}_sample_ids.txt", "w", encoding="utf-8") as f:
        for sample_id in sorted(selected_ids):
            f.write(sample_id + "\n")

    selected_df = pd.DataFrame([item.__dict__ for item in selected])
    selected_df.to_csv(report_dir / f"{split}_selected_metadata.csv", index=False)
    report = {
        "target": target,
        "selected": len(selected_ids),
        "exported_rows": exported_rows,
        "unique_mmsi": int(selected_df["mmsi"].nunique()) if not selected_df.empty else 0,
        "unique_segments": int(selected_df["segment_id"].nunique()) if not selected_df.empty else 0,
        "ship_group_counts": selected_df["ship_group"].value_counts().to_dict() if not selected_df.empty else {},
        "speed_bin_counts": selected_df["speed_bin"].value_counts().to_dict() if not selected_df.empty else {},
        "motion_bin_counts": selected_df["motion_bin"].value_counts().to_dict() if not selected_df.empty else {},
        "turn_bin_counts": selected_df["turn_bin"].value_counts().to_dict() if not selected_df.empty else {},
        "strata_report": strata_report,
    }
    with open(report_dir / f"{split}_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)

    overall = {}
    for split, target in [
        ("train", args.train_target),
        ("val", args.val_target),
        ("test", args.test_target),
    ]:
        print(f"[build] split={split} target={target}")
        overall[split] = build_split(
            split=split,
            target=target,
            benchmark_root=args.benchmark_root,
            output_root=args.output_root,
            chunk_size=args.chunk_size,
            oversample_factor=args.oversample_factor,
            max_per_mmsi=args.max_per_mmsi,
            max_per_segment=args.max_per_segment,
            min_segment_step_gap=args.min_segment_step_gap,
        )
        print(json.dumps(overall[split], indent=2))

    with open(args.output_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)
    print("[build] finished")


if __name__ == "__main__":
    main()
