#!/usr/bin/env python3
"""Build a cleaner and more predictable Ship-Core-Lite benchmark subset.

This builder is intentionally quality-first. It prefers smoother, more
continuous, lower-noise trajectories from regular vessel groups and avoids
 aggressively rebalancing difficult long-tail strata.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


KNOTS_PER_MPS = 1.9438444924406
STEP_SECONDS = 20.0

DEFAULT_BENCHMARK_ROOT = Path(
    "/mnt/nfs/kun/DeepJSCC/ship_trajectory_datesets/benchmark/core"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/nfs/kun/DeepJSCC/ship_trajectory_datesets/mini_benchmark/clean_ship_core_lite_v1"
)


@dataclass
class Candidate:
    sample_id: str
    hash_value: int
    split: str
    mmsi: int
    segment_id: str
    sample_step: int
    ship_group: str
    speed_bin: str
    motion_bin: str
    quality_score: float
    difficulty_score: float
    avg_speed_knots: float
    hist_median_speed_knots: float
    hist_speed_cv: float
    hist_efficiency: float
    fut_efficiency: float
    future_linearity: float
    hist_nonzero_ratio: float
    fut_nonzero_ratio: float
    hist_pause_ratio: float
    fut_pause_ratio: float
    bridge_turn_deg: float
    hist_turn_mean_abs_deg: float
    fut_turn_mean_abs_deg: float
    hist_turn_max_abs_deg: float
    fut_turn_max_abs_deg: float
    hist_step_outlier_ratio: float
    fut_step_outlier_ratio: float
    interp_ratio_hist: float
    interp_ratio_fut: float
    interp_ratio_total: float
    hist_interp_true_ratio: float
    fut_interp_true_ratio: float
    hist_cog_motion_mae_deg: float
    hist_heading_cog_mae_deg: float
    hist_displacement_m: float
    fut_displacement_m: float
    future_path_length_m: float
    hard_filter_margin: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a high-quality, low-noise, predictable clean ship benchmark."
    )
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--train-target", type=int, default=19500)
    parser.add_argument("--val-target", type=int, default=2400)
    parser.add_argument("--test-target", type=int, default=2100)
    parser.add_argument("--allow-ship-groups", default="cargo_tanker,passenger_ferry")
    parser.add_argument(
        "--exclude-ship-groups",
        default="other_unknown,fishing,tug_service,sailing_leisure",
    )
    parser.add_argument("--max-per-mmsi", type=int, default=12)
    parser.add_argument("--max-per-segment", type=int, default=3)
    parser.add_argument("--min-segment-step-gap", type=int, default=20)
    parser.add_argument("--quota-exponent", type=float, default=0.92)
    parser.add_argument("--log-every", type=int, default=50000)
    parser.add_argument("--min-avg-speed-knots", type=float, default=3.5)
    parser.add_argument("--max-avg-speed-knots", type=float, default=22.0)
    parser.add_argument("--min-hist-median-speed-knots", type=float, default=4.0)
    parser.add_argument("--max-hist-speed-cv", type=float, default=0.35)
    parser.add_argument("--min-hist-efficiency", type=float, default=0.9)
    parser.add_argument("--min-fut-efficiency", type=float, default=0.94)
    parser.add_argument("--min-future-linearity", type=float, default=0.94)
    parser.add_argument("--min-hist-nonzero-ratio", type=float, default=0.85)
    parser.add_argument("--min-fut-nonzero-ratio", type=float, default=0.9)
    parser.add_argument("--max-hist-pause-ratio", type=float, default=0.18)
    parser.add_argument("--max-fut-pause-ratio", type=float, default=0.15)
    parser.add_argument("--max-bridge-turn-deg", type=float, default=30.0)
    parser.add_argument("--max-hist-turn-mean-deg", type=float, default=10.0)
    parser.add_argument("--max-fut-turn-mean-deg", type=float, default=8.0)
    parser.add_argument("--max-hist-turn-max-deg", type=float, default=35.0)
    parser.add_argument("--max-fut-turn-max-deg", type=float, default=28.0)
    parser.add_argument("--max-step-outlier-ratio", type=float, default=3.0)
    parser.add_argument("--max-interp-ratio-total", type=float, default=0.12)
    parser.add_argument("--max-interp-ratio-side", type=float, default=0.1)
    parser.add_argument("--max-interp-true-ratio", type=float, default=0.12)
    parser.add_argument("--max-hist-cog-motion-mae-deg", type=float, default=22.0)
    parser.add_argument("--max-hist-heading-cog-mae-deg", type=float, default=20.0)
    parser.add_argument("--min-quality-score", type=float, default=60.0)
    return parser.parse_args()


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: object, default: int = 0) -> int:
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except Exception:
        return default


def safe_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    text = normalize_text(value)
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return default


def parse_json_array(text: str) -> list[object]:
    text = str(text or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except Exception:
        return []
    return data if isinstance(data, list) else []


def parse_float_array(text: str) -> list[float]:
    values = []
    for item in parse_json_array(text):
        try:
            values.append(float(item))
        except Exception:
            continue
    return values


def parse_bool_array(text: str) -> list[bool]:
    return [bool(item) for item in parse_json_array(text)]


def stable_hash(sample_id: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(sample_id.encode("utf-8"), digest_size=8).digest(),
        byteorder="big",
    )


def parse_sample_step(sample_id: str) -> int:
    try:
        return int(sample_id.rsplit("_", 1)[-1])
    except Exception:
        return -1


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
    hist_speed = hist_disp_m / 600.0 * KNOTS_PER_MPS
    fut_speed = fut_disp_m / 600.0 * KNOTS_PER_MPS
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
    if fut_disp_m < 700:
        return "low"
    if fut_disp_m < 1700:
        return "mid"
    return "high"


def wrap_angle_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def wrap_angle_diff_deg(a: float, b: float) -> float:
    return abs(wrap_angle_deg(b - a))


def angle_from_sin_cos(sin_values: list[float], cos_values: list[float]) -> list[float]:
    count = min(len(sin_values), len(cos_values))
    angles = []
    for idx in range(count):
        angles.append(math.degrees(math.atan2(sin_values[idx], cos_values[idx])))
    return angles


def points_from_xy(x_values: list[float], y_values: list[float]) -> list[tuple[float, float]]:
    count = min(len(x_values), len(y_values))
    return [(float(x_values[i]), float(y_values[i])) for i in range(count)]


def distance(p0: tuple[float, float], p1: tuple[float, float]) -> float:
    return math.hypot(p1[0] - p0[0], p1[1] - p0[1])


def median(values: Iterable[float], default: float = 0.0) -> float:
    values_list = list(values)
    if not values_list:
        return default
    return float(statistics.median(values_list))


def mean(values: Iterable[float], default: float = 0.0) -> float:
    values_list = list(values)
    if not values_list:
        return default
    return float(sum(values_list) / len(values_list))


def pstdev(values: Iterable[float], default: float = 0.0) -> float:
    values_list = list(values)
    if len(values_list) < 2:
        return default
    try:
        return float(statistics.pstdev(values_list))
    except Exception:
        return default


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def gaussian_preference(value: float, center: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return math.exp(-((value - center) ** 2) / (2.0 * scale * scale))


def compute_path_metrics(
    points: list[tuple[float, float]],
    prepend_origin: bool,
) -> dict[str, float]:
    if prepend_origin:
        seq = [(0.0, 0.0)] + points
    else:
        seq = points
    if len(seq) < 2:
        return {
            "path_length_m": 0.0,
            "displacement_m": 0.0,
            "efficiency": 0.0,
            "median_speed_knots": 0.0,
            "speed_cv": 0.0,
            "nonzero_ratio": 0.0,
            "pause_ratio": 1.0,
            "mean_abs_turn_deg": 180.0,
            "max_abs_turn_deg": 180.0,
            "step_outlier_ratio": 999.0,
            "recent_bearing_deg": 0.0,
            "turn_count": 0.0,
        }

    step_lengths = []
    step_bearings = []
    for idx in range(1, len(seq)):
        p0 = seq[idx - 1]
        p1 = seq[idx]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        step_lengths.append(math.hypot(dx, dy))
        step_bearings.append(math.degrees(math.atan2(dy, dx)))

    if prepend_origin:
        displacement_m = math.hypot(points[-1][0], points[-1][1]) if points else 0.0
    else:
        displacement_m = distance(points[0], points[-1]) if len(points) >= 2 else 0.0
    path_length_m = sum(step_lengths)
    efficiency = displacement_m / path_length_m if path_length_m > 1e-6 else 0.0
    speed_knots = [step / STEP_SECONDS * KNOTS_PER_MPS for step in step_lengths]
    median_speed_knots = median(speed_knots)
    mean_speed_knots = mean(speed_knots)
    speed_cv = pstdev(speed_knots) / mean_speed_knots if mean_speed_knots > 1e-6 else 0.0
    nonzero_ratio = sum(1 for step in step_lengths if step >= 5.0) / len(step_lengths)
    pause_ratio = sum(1 for step in step_lengths if step <= 10.0) / len(step_lengths)
    turns = [
        wrap_angle_diff_deg(step_bearings[idx - 1], step_bearings[idx])
        for idx in range(1, len(step_bearings))
    ]
    mean_abs_turn_deg = mean(turns)
    max_abs_turn_deg = max(turns) if turns else 0.0
    median_step = max(median(step_lengths, default=0.0), 1.0)
    step_outlier_ratio = (max(step_lengths) / median_step) if step_lengths else 999.0
    recent_bearing_deg = mean_angle_deg(step_bearings[-5:]) if step_bearings else 0.0
    return {
        "path_length_m": path_length_m,
        "displacement_m": displacement_m,
        "efficiency": efficiency,
        "median_speed_knots": median_speed_knots,
        "speed_cv": speed_cv,
        "nonzero_ratio": nonzero_ratio,
        "pause_ratio": pause_ratio,
        "mean_abs_turn_deg": mean_abs_turn_deg,
        "max_abs_turn_deg": max_abs_turn_deg,
        "step_outlier_ratio": step_outlier_ratio,
        "recent_bearing_deg": recent_bearing_deg,
        "turn_count": float(len(turns)),
    }


def mean_angle_deg(angles: Iterable[float]) -> float:
    angles_list = list(angles)
    if not angles_list:
        return 0.0
    sin_sum = sum(math.sin(math.radians(v)) for v in angles_list)
    cos_sum = sum(math.cos(math.radians(v)) for v in angles_list)
    if abs(sin_sum) < 1e-9 and abs(cos_sum) < 1e-9:
        return float(angles_list[-1])
    return math.degrees(math.atan2(sin_sum, cos_sum))


def mean_abs_angle_error(a_values: list[float], b_values: list[float]) -> float:
    count = min(len(a_values), len(b_values))
    if count == 0:
        return 0.0
    return mean(wrap_angle_diff_deg(a_values[idx], b_values[idx]) for idx in range(count))


def recent_motion_bearings(points: list[tuple[float, float]], count: int = 6) -> list[float]:
    if len(points) < 2:
        return []
    bearings = []
    start = max(1, len(points) - count)
    for idx in range(start, len(points)):
        p0 = points[idx - 1]
        p1 = points[idx]
        if distance(p0, p1) < 1e-6:
            continue
        bearings.append(math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0])))
    return bearings


def compute_quality_score(metrics: dict[str, float], ship_group: str) -> tuple[float, float]:
    group_bonus = 6.0 if ship_group == "cargo_tanker" else 4.5
    moderate_speed = gaussian_preference(metrics["avg_speed_knots"], center=12.0, scale=4.0)
    quality_score = (
        14.0 * metrics["fut_efficiency"]
        + 10.0 * metrics["hist_efficiency"]
        + 9.0 * metrics["future_linearity"]
        + 8.0 * clamp01(1.0 - metrics["hist_speed_cv"] / 0.35)
        + 8.0 * clamp01(1.0 - metrics["fut_turn_mean_abs_deg"] / 8.0)
        + 6.0 * clamp01(1.0 - metrics["hist_turn_mean_abs_deg"] / 10.0)
        + 6.0 * clamp01(1.0 - metrics["bridge_turn_deg"] / 30.0)
        + 5.0 * clamp01(1.0 - metrics["hist_pause_ratio"] / 0.18)
        + 5.0 * clamp01(1.0 - metrics["fut_pause_ratio"] / 0.15)
        + 5.0 * clamp01(1.0 - metrics["interp_ratio_total"] / 0.12)
        + 4.0 * clamp01(1.0 - metrics["hist_step_outlier_ratio"] / 3.0)
        + 4.0 * clamp01(1.0 - metrics["fut_step_outlier_ratio"] / 3.0)
        + 4.0 * clamp01(1.0 - metrics["hist_cog_motion_mae_deg"] / 22.0)
        + 4.0 * clamp01(1.0 - metrics["hist_heading_cog_mae_deg"] / 20.0)
        + 6.0 * moderate_speed
        + group_bonus
    )
    difficulty_score = (
        30.0 * (1.0 - metrics["fut_efficiency"])
        + 20.0 * clamp01(metrics["fut_turn_mean_abs_deg"] / 20.0)
        + 15.0 * clamp01(metrics["bridge_turn_deg"] / 45.0)
        + 15.0 * metrics["fut_pause_ratio"]
        + 10.0 * clamp01(metrics["fut_step_outlier_ratio"] / 4.0)
        + 10.0 * clamp01(metrics["interp_ratio_total"] / 0.2)
    )
    return quality_score, difficulty_score


def quality_margin(metrics: dict[str, float], args: argparse.Namespace) -> float:
    margins = [
        metrics["avg_speed_knots"] - args.min_avg_speed_knots,
        args.max_avg_speed_knots - metrics["avg_speed_knots"],
        metrics["hist_median_speed_knots"] - args.min_hist_median_speed_knots,
        args.max_hist_speed_cv - metrics["hist_speed_cv"],
        metrics["hist_efficiency"] - args.min_hist_efficiency,
        metrics["fut_efficiency"] - args.min_fut_efficiency,
        metrics["future_linearity"] - args.min_future_linearity,
        metrics["hist_nonzero_ratio"] - args.min_hist_nonzero_ratio,
        metrics["fut_nonzero_ratio"] - args.min_fut_nonzero_ratio,
        args.max_hist_pause_ratio - metrics["hist_pause_ratio"],
        args.max_fut_pause_ratio - metrics["fut_pause_ratio"],
        args.max_bridge_turn_deg - metrics["bridge_turn_deg"],
        args.max_hist_turn_mean_deg - metrics["hist_turn_mean_abs_deg"],
        args.max_fut_turn_mean_deg - metrics["fut_turn_mean_abs_deg"],
        args.max_hist_turn_max_deg - metrics["hist_turn_max_abs_deg"],
        args.max_fut_turn_max_deg - metrics["fut_turn_max_abs_deg"],
        args.max_step_outlier_ratio - metrics["hist_step_outlier_ratio"],
        args.max_step_outlier_ratio - metrics["fut_step_outlier_ratio"],
        args.max_interp_ratio_total - metrics["interp_ratio_total"],
        args.max_interp_ratio_side - metrics["interp_ratio_hist"],
        args.max_interp_ratio_side - metrics["interp_ratio_fut"],
        args.max_interp_true_ratio - metrics["hist_interp_true_ratio"],
        args.max_interp_true_ratio - metrics["fut_interp_true_ratio"],
        args.max_hist_cog_motion_mae_deg - metrics["hist_cog_motion_mae_deg"],
        args.max_hist_heading_cog_mae_deg - metrics["hist_heading_cog_mae_deg"],
    ]
    return min(margins)


def check_metric_filters(metrics: dict[str, float], args: argparse.Namespace) -> str | None:
    checks = [
        ("avg_speed_out_of_range", args.min_avg_speed_knots <= metrics["avg_speed_knots"] <= args.max_avg_speed_knots),
        ("low_hist_median_speed", metrics["hist_median_speed_knots"] >= args.min_hist_median_speed_knots),
        ("hist_speed_variability", metrics["hist_speed_cv"] <= args.max_hist_speed_cv),
        ("hist_efficiency", metrics["hist_efficiency"] >= args.min_hist_efficiency),
        ("fut_efficiency", metrics["fut_efficiency"] >= args.min_fut_efficiency),
        ("future_linearity", metrics["future_linearity"] >= args.min_future_linearity),
        ("hist_nonzero_ratio", metrics["hist_nonzero_ratio"] >= args.min_hist_nonzero_ratio),
        ("fut_nonzero_ratio", metrics["fut_nonzero_ratio"] >= args.min_fut_nonzero_ratio),
        ("hist_pause_ratio", metrics["hist_pause_ratio"] <= args.max_hist_pause_ratio),
        ("fut_pause_ratio", metrics["fut_pause_ratio"] <= args.max_fut_pause_ratio),
        ("bridge_turn", metrics["bridge_turn_deg"] <= args.max_bridge_turn_deg),
        ("hist_turn_mean", metrics["hist_turn_mean_abs_deg"] <= args.max_hist_turn_mean_deg),
        ("fut_turn_mean", metrics["fut_turn_mean_abs_deg"] <= args.max_fut_turn_mean_deg),
        ("hist_turn_max", metrics["hist_turn_max_abs_deg"] <= args.max_hist_turn_max_deg),
        ("fut_turn_max", metrics["fut_turn_max_abs_deg"] <= args.max_fut_turn_max_deg),
        (
            "step_outlier_ratio",
            metrics["hist_step_outlier_ratio"] <= args.max_step_outlier_ratio
            and metrics["fut_step_outlier_ratio"] <= args.max_step_outlier_ratio,
        ),
        ("interp_ratio_total", metrics["interp_ratio_total"] <= args.max_interp_ratio_total),
        (
            "interp_ratio_side",
            metrics["interp_ratio_hist"] <= args.max_interp_ratio_side
            and metrics["interp_ratio_fut"] <= args.max_interp_ratio_side,
        ),
        (
            "interp_true_ratio",
            metrics["hist_interp_true_ratio"] <= args.max_interp_true_ratio
            and metrics["fut_interp_true_ratio"] <= args.max_interp_true_ratio,
        ),
        ("hist_cog_motion_mae", metrics["hist_cog_motion_mae_deg"] <= args.max_hist_cog_motion_mae_deg),
        ("hist_heading_cog_mae", metrics["hist_heading_cog_mae_deg"] <= args.max_hist_heading_cog_mae_deg),
    ]
    for name, ok in checks:
        if not ok:
            return name
    return None


def evaluate_row(
    row: dict[str, str],
    split: str,
    args: argparse.Namespace,
    allowed_groups: set[str],
    excluded_groups: set[str],
) -> tuple[Candidate | None, str | None]:
    ship_group = ship_group_from_type(row.get("ship_type"), row.get("ship_class"))
    if ship_group in excluded_groups or ship_group not in allowed_groups:
        return None, "ship_group_excluded"
    if "core_eligible" in row and not safe_bool(row.get("core_eligible"), default=True):
        return None, "not_core_eligible"
    if "quality_tier" in row and normalize_text(row.get("quality_tier")) not in {"", "core"}:
        return None, "quality_tier_not_core"
    if "hist_gap_ok" in row and not safe_bool(row.get("hist_gap_ok"), default=True):
        return None, "hist_gap_not_ok"
    if "fut_gap_ok" in row and not safe_bool(row.get("fut_gap_ok"), default=True):
        return None, "fut_gap_not_ok"

    hist_x = parse_float_array(row.get("hist_x_json", ""))
    hist_y = parse_float_array(row.get("hist_y_json", ""))
    fut_x = parse_float_array(row.get("fut_x_json", ""))
    fut_y = parse_float_array(row.get("fut_y_json", ""))
    hist_points = points_from_xy(hist_x, hist_y)
    fut_points = points_from_xy(fut_x, fut_y)
    if len(hist_points) < 12 or len(fut_points) < 12:
        return None, "too_few_points"

    hist_sog = parse_float_array(row.get("hist_sog_json", ""))
    hist_cog = angle_from_sin_cos(
        parse_float_array(row.get("hist_cog_sin_json", "")),
        parse_float_array(row.get("hist_cog_cos_json", "")),
    )
    hist_heading = angle_from_sin_cos(
        parse_float_array(row.get("hist_heading_sin_json", "")),
        parse_float_array(row.get("hist_heading_cos_json", "")),
    )
    hist_interp = parse_bool_array(row.get("hist_interp_json", ""))
    fut_interp = parse_bool_array(row.get("fut_interp_json", ""))

    hist_metrics = compute_path_metrics(hist_points, prepend_origin=False)
    fut_metrics = compute_path_metrics(fut_points, prepend_origin=True)

    hist_motion_bearings = recent_motion_bearings(hist_points)
    recent_hist_motion = mean_angle_deg(hist_motion_bearings[-5:]) if hist_motion_bearings else hist_metrics["recent_bearing_deg"]
    recent_hist_cog = mean_angle_deg(hist_cog[-5:]) if hist_cog else recent_hist_motion
    bridge_turn_deg = wrap_angle_diff_deg(recent_hist_motion, fut_metrics["recent_bearing_deg"])
    hist_cog_motion_mae_deg = (
        mean_abs_angle_error(hist_cog[-len(hist_motion_bearings):], hist_motion_bearings)
        if hist_motion_bearings and hist_cog
        else 0.0
    )
    hist_heading_cog_mae_deg = mean_abs_angle_error(hist_heading, hist_cog) if hist_heading and hist_cog else 0.0
    hist_interp_true_ratio = mean(1.0 if flag else 0.0 for flag in hist_interp) if hist_interp else 0.0
    fut_interp_true_ratio = mean(1.0 if flag else 0.0 for flag in fut_interp) if fut_interp else 0.0

    hist_displacement_m = safe_float(row.get("hist_displacement_m"), hist_metrics["displacement_m"])
    fut_displacement_m = safe_float(row.get("fut_displacement_m"), fut_metrics["displacement_m"])
    interp_ratio_hist = safe_float(row.get("interp_ratio_hist"), hist_interp_true_ratio)
    interp_ratio_fut = safe_float(row.get("interp_ratio_fut"), fut_interp_true_ratio)
    interp_ratio_total = safe_float(
        row.get("interp_ratio_total"),
        0.5 * (interp_ratio_hist + interp_ratio_fut),
    )
    avg_speed = avg_speed_knots(hist_displacement_m, fut_displacement_m)

    hist_sog_median = median(hist_sog) if hist_sog else hist_metrics["median_speed_knots"]
    future_linearity = fut_displacement_m / max(fut_metrics["path_length_m"], 1.0)

    metrics = {
        "avg_speed_knots": avg_speed,
        "hist_median_speed_knots": hist_sog_median,
        "hist_speed_cv": hist_metrics["speed_cv"],
        "hist_efficiency": hist_metrics["efficiency"],
        "fut_efficiency": fut_metrics["efficiency"],
        "future_linearity": future_linearity,
        "hist_nonzero_ratio": hist_metrics["nonzero_ratio"],
        "fut_nonzero_ratio": fut_metrics["nonzero_ratio"],
        "hist_pause_ratio": hist_metrics["pause_ratio"],
        "fut_pause_ratio": fut_metrics["pause_ratio"],
        "bridge_turn_deg": bridge_turn_deg,
        "hist_turn_mean_abs_deg": hist_metrics["mean_abs_turn_deg"],
        "fut_turn_mean_abs_deg": fut_metrics["mean_abs_turn_deg"],
        "hist_turn_max_abs_deg": hist_metrics["max_abs_turn_deg"],
        "fut_turn_max_abs_deg": fut_metrics["max_abs_turn_deg"],
        "hist_step_outlier_ratio": hist_metrics["step_outlier_ratio"],
        "fut_step_outlier_ratio": fut_metrics["step_outlier_ratio"],
        "interp_ratio_hist": interp_ratio_hist,
        "interp_ratio_fut": interp_ratio_fut,
        "interp_ratio_total": interp_ratio_total,
        "hist_interp_true_ratio": hist_interp_true_ratio,
        "fut_interp_true_ratio": fut_interp_true_ratio,
        "hist_cog_motion_mae_deg": hist_cog_motion_mae_deg,
        "hist_heading_cog_mae_deg": hist_heading_cog_mae_deg,
    }

    filter_reason = check_metric_filters(metrics, args)
    if filter_reason is not None:
        return None, filter_reason

    quality_score, difficulty_score = compute_quality_score(metrics, ship_group)
    if quality_score < args.min_quality_score:
        return None, "quality_score"

    sample_id = str(row["sample_id"])
    candidate = Candidate(
        sample_id=sample_id,
        hash_value=stable_hash(sample_id),
        split=split,
        mmsi=safe_int(row.get("mmsi")),
        segment_id=str(row.get("segment_id", "")),
        sample_step=parse_sample_step(sample_id),
        ship_group=ship_group,
        speed_bin=speed_bin_from_knots(avg_speed),
        motion_bin=motion_bin_from_disp(fut_displacement_m),
        quality_score=quality_score,
        difficulty_score=difficulty_score,
        avg_speed_knots=avg_speed,
        hist_median_speed_knots=hist_sog_median,
        hist_speed_cv=hist_metrics["speed_cv"],
        hist_efficiency=hist_metrics["efficiency"],
        fut_efficiency=fut_metrics["efficiency"],
        future_linearity=future_linearity,
        hist_nonzero_ratio=hist_metrics["nonzero_ratio"],
        fut_nonzero_ratio=fut_metrics["nonzero_ratio"],
        hist_pause_ratio=hist_metrics["pause_ratio"],
        fut_pause_ratio=fut_metrics["pause_ratio"],
        bridge_turn_deg=bridge_turn_deg,
        hist_turn_mean_abs_deg=hist_metrics["mean_abs_turn_deg"],
        fut_turn_mean_abs_deg=fut_metrics["mean_abs_turn_deg"],
        hist_turn_max_abs_deg=hist_metrics["max_abs_turn_deg"],
        fut_turn_max_abs_deg=fut_metrics["max_abs_turn_deg"],
        hist_step_outlier_ratio=hist_metrics["step_outlier_ratio"],
        fut_step_outlier_ratio=fut_metrics["step_outlier_ratio"],
        interp_ratio_hist=interp_ratio_hist,
        interp_ratio_fut=interp_ratio_fut,
        interp_ratio_total=interp_ratio_total,
        hist_interp_true_ratio=hist_interp_true_ratio,
        fut_interp_true_ratio=fut_interp_true_ratio,
        hist_cog_motion_mae_deg=hist_cog_motion_mae_deg,
        hist_heading_cog_mae_deg=hist_heading_cog_mae_deg,
        hist_displacement_m=hist_displacement_m,
        fut_displacement_m=fut_displacement_m,
        future_path_length_m=fut_metrics["path_length_m"],
        hard_filter_margin=quality_margin(metrics, args),
    )
    return candidate, None


def weighted_quota(
    counts: dict[tuple[str, str, str], int],
    target: int,
    quota_exponent: float,
) -> dict[tuple[str, str, str], int]:
    if target <= 0 or not counts:
        return {key: 0 for key in counts}
    speed_weights = {"0_5": 0.45, "5_10": 1.0, "10_15": 1.0, "15_20": 0.82, "gt20": 0.55}
    motion_weights = {"low": 0.8, "mid": 1.0, "high": 0.72}
    group_weights = {"cargo_tanker": 1.0, "passenger_ferry": 0.9}

    raw_weights: dict[tuple[str, str, str], float] = {}
    for key, count in counts.items():
        ship_group, speed_bin, motion_bin = key
        raw_weights[key] = (
            (count ** quota_exponent)
            * group_weights.get(ship_group, 0.5)
            * speed_weights.get(speed_bin, 0.75)
            * motion_weights.get(motion_bin, 0.8)
        )

    total_weight = sum(raw_weights.values())
    if total_weight <= 0:
        return {key: 0 for key in counts}

    quota = {key: 0 for key in counts}
    fractional = []
    remaining = target
    for key, weight in raw_weights.items():
        exact = target * weight / total_weight
        base = min(counts[key], int(math.floor(exact)))
        quota[key] = base
        remaining -= base
        fractional.append((exact - base, counts[key] - base, key))

    for _, _, key in sorted(fractional, reverse=True):
        if remaining <= 0:
            break
        if quota[key] < counts[key]:
            quota[key] += 1
            remaining -= 1
    return quota


def coarse_key(candidate: Candidate) -> tuple[str, str, str]:
    return (candidate.ship_group, candidate.speed_bin, candidate.motion_bin)


def can_take(
    candidate: Candidate,
    used_mmsi: Counter[int],
    used_segment: Counter[str],
    used_steps: defaultdict[str, list[int]],
    args: argparse.Namespace,
) -> bool:
    if used_mmsi[candidate.mmsi] >= args.max_per_mmsi:
        return False
    if used_segment[candidate.segment_id] >= args.max_per_segment:
        return False
    if candidate.sample_step >= 0:
        for previous in used_steps[candidate.segment_id]:
            if abs(previous - candidate.sample_step) < args.min_segment_step_gap:
                return False
    return True


def select_candidates(
    split: str,
    candidates: list[Candidate],
    target: int,
    args: argparse.Namespace,
) -> tuple[list[Candidate], dict[str, object]]:
    by_key: dict[tuple[str, str, str], list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_key[coarse_key(candidate)].append(candidate)
    for values in by_key.values():
        values.sort(key=lambda item: (-item.quality_score, item.difficulty_score, item.hash_value))

    counts = {key: len(values) for key, values in by_key.items()}
    quota = weighted_quota(counts, target, args.quota_exponent)
    selected: list[Candidate] = []
    leftovers: list[Candidate] = []
    used_mmsi: Counter[int] = Counter()
    used_segment: Counter[str] = Counter()
    used_steps: defaultdict[str, list[int]] = defaultdict(list)
    strata_report: dict[str, dict[str, object]] = {}

    for key, pool in sorted(by_key.items()):
        taken = 0
        wanted = quota.get(key, 0)
        for candidate in pool:
            if taken < wanted and can_take(candidate, used_mmsi, used_segment, used_steps, args):
                selected.append(candidate)
                used_mmsi[candidate.mmsi] += 1
                used_segment[candidate.segment_id] += 1
                if candidate.sample_step >= 0:
                    used_steps[candidate.segment_id].append(candidate.sample_step)
                taken += 1
            else:
                leftovers.append(candidate)
        strata_report["|".join(key)] = {
            "eligible": len(pool),
            "quota": wanted,
            "selected": taken,
            "mean_quality_score": round(mean(item.quality_score for item in pool), 4),
            "mean_difficulty_score": round(mean(item.difficulty_score for item in pool), 4),
        }

    leftovers.sort(key=lambda item: (-item.quality_score, item.difficulty_score, item.hash_value))
    fill_added = 0
    for candidate in leftovers:
        if len(selected) >= target:
            break
        if not can_take(candidate, used_mmsi, used_segment, used_steps, args):
            continue
        selected.append(candidate)
        used_mmsi[candidate.mmsi] += 1
        used_segment[candidate.segment_id] += 1
        if candidate.sample_step >= 0:
            used_steps[candidate.segment_id].append(candidate.sample_step)
        fill_added += 1

    selected.sort(key=lambda item: (-item.quality_score, item.difficulty_score, item.hash_value))
    selected = selected[:target]
    report = {
        "split": split,
        "target": target,
        "eligible_candidates": len(candidates),
        "selected": len(selected),
        "fill_added": fill_added,
        "strata_report": strata_report,
    }
    return selected, report


def export_selected_rows(
    split_dir: Path,
    selected_ids: set[str],
    output_path: Path,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    writer: csv.DictWriter | None = None
    with gzip.open(output_path, "wt", encoding="utf-8", newline="") as dst:
        for file_path in sorted(split_dir.glob("*.csv.gz")):
            with gzip.open(file_path, "rt", encoding="utf-8", newline="") as src:
                reader = csv.DictReader(src)
                if writer is None:
                    writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
                    writer.writeheader()
                for row in reader:
                    if row.get("sample_id") not in selected_ids:
                        continue
                    writer.writerow(row)
                    kept += 1
    return kept


def write_metadata_csv(path: Path, candidates: list[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(candidates[0]).keys()) if candidates else list(asdict(Candidate(
        sample_id="",
        hash_value=0,
        split="",
        mmsi=0,
        segment_id="",
        sample_step=0,
        ship_group="",
        speed_bin="",
        motion_bin="",
        quality_score=0.0,
        difficulty_score=0.0,
        avg_speed_knots=0.0,
        hist_median_speed_knots=0.0,
        hist_speed_cv=0.0,
        hist_efficiency=0.0,
        fut_efficiency=0.0,
        future_linearity=0.0,
        hist_nonzero_ratio=0.0,
        fut_nonzero_ratio=0.0,
        hist_pause_ratio=0.0,
        fut_pause_ratio=0.0,
        bridge_turn_deg=0.0,
        hist_turn_mean_abs_deg=0.0,
        fut_turn_mean_abs_deg=0.0,
        hist_turn_max_abs_deg=0.0,
        fut_turn_max_abs_deg=0.0,
        hist_step_outlier_ratio=0.0,
        fut_step_outlier_ratio=0.0,
        interp_ratio_hist=0.0,
        interp_ratio_fut=0.0,
        interp_ratio_total=0.0,
        hist_interp_true_ratio=0.0,
        fut_interp_true_ratio=0.0,
        hist_cog_motion_mae_deg=0.0,
        hist_heading_cog_mae_deg=0.0,
        hist_displacement_m=0.0,
        fut_displacement_m=0.0,
        future_path_length_m=0.0,
        hard_filter_margin=0.0,
    )).keys())
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(asdict(candidate))


def summarize_numeric(candidates: list[Candidate], attr: str) -> dict[str, float]:
    values = [float(getattr(item, attr)) for item in candidates]
    if not values:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    ordered = sorted(values)
    p90_idx = min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))
    return {
        "mean": round(mean(values), 4),
        "median": round(median(values), 4),
        "p90": round(ordered[p90_idx], 4),
        "min": round(ordered[0], 4),
        "max": round(ordered[-1], 4),
    }


def write_report(
    path: Path,
    split: str,
    target: int,
    selected: list[Candidate],
    scan_stats: dict[str, object],
    selection_report: dict[str, object],
    exported_rows: int,
) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    ship_group_counts = Counter(item.ship_group for item in selected)
    speed_bin_counts = Counter(item.speed_bin for item in selected)
    motion_bin_counts = Counter(item.motion_bin for item in selected)
    report = {
        "split": split,
        "target": target,
        "selected": len(selected),
        "exported_rows": exported_rows,
        "unique_mmsi": len({item.mmsi for item in selected}),
        "unique_segments": len({item.segment_id for item in selected}),
        "ship_group_counts": dict(ship_group_counts),
        "speed_bin_counts": dict(speed_bin_counts),
        "motion_bin_counts": dict(motion_bin_counts),
        "filter_counts": scan_stats["filter_counts"],
        "input_rows": scan_stats["input_rows"],
        "eligible_candidates": scan_stats["eligible_candidates"],
        "selection_report": selection_report,
        "quality_stats": {
            "quality_score": summarize_numeric(selected, "quality_score"),
            "difficulty_score": summarize_numeric(selected, "difficulty_score"),
            "avg_speed_knots": summarize_numeric(selected, "avg_speed_knots"),
            "hist_speed_cv": summarize_numeric(selected, "hist_speed_cv"),
            "hist_efficiency": summarize_numeric(selected, "hist_efficiency"),
            "fut_efficiency": summarize_numeric(selected, "fut_efficiency"),
            "future_linearity": summarize_numeric(selected, "future_linearity"),
            "bridge_turn_deg": summarize_numeric(selected, "bridge_turn_deg"),
            "hist_turn_mean_abs_deg": summarize_numeric(selected, "hist_turn_mean_abs_deg"),
            "fut_turn_mean_abs_deg": summarize_numeric(selected, "fut_turn_mean_abs_deg"),
            "interp_ratio_total": summarize_numeric(selected, "interp_ratio_total"),
            "hard_filter_margin": summarize_numeric(selected, "hard_filter_margin"),
        },
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def scan_split(
    split: str,
    split_dir: Path,
    args: argparse.Namespace,
    allowed_groups: set[str],
    excluded_groups: set[str],
) -> tuple[list[Candidate], dict[str, object]]:
    candidates: list[Candidate] = []
    filter_counts: Counter[str] = Counter()
    input_rows = 0
    for file_idx, file_path in enumerate(sorted(split_dir.glob("*.csv.gz")), start=1):
        print(f"[scan] {split}: file {file_idx} {file_path.name}")
        with gzip.open(file_path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                input_rows += 1
                candidate, reason = evaluate_row(row, split, args, allowed_groups, excluded_groups)
                if candidate is None:
                    filter_counts[reason or "unknown"] += 1
                else:
                    candidates.append(candidate)
                if input_rows % args.log_every == 0:
                    print(
                        f"[scan] {split}: processed={input_rows} "
                        f"eligible={len(candidates)} filtered={input_rows - len(candidates)}"
                    )
    candidates.sort(key=lambda item: (-item.quality_score, item.difficulty_score, item.hash_value))
    stats = {
        "input_rows": input_rows,
        "eligible_candidates": len(candidates),
        "filter_counts": dict(filter_counts),
    }
    print(
        f"[scan] {split}: finished input_rows={input_rows} "
        f"eligible={len(candidates)} filtered={input_rows - len(candidates)}"
    )
    return candidates, stats


def build_split(
    split: str,
    target: int,
    benchmark_root: Path,
    output_root: Path,
    args: argparse.Namespace,
    allowed_groups: set[str],
    excluded_groups: set[str],
) -> dict[str, object]:
    split_dir = benchmark_root / split
    candidates, scan_stats = scan_split(split, split_dir, args, allowed_groups, excluded_groups)
    selected, selection_report = select_candidates(split, candidates, target, args)
    selected_ids = {item.sample_id for item in selected}

    split_out_dir = output_root / split
    sample_id_dir = output_root / "sample_ids"
    report_dir = output_root / "reports"
    split_out_dir.mkdir(parents=True, exist_ok=True)
    sample_id_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    output_csv = split_out_dir / "part-000.csv.gz"
    exported_rows = export_selected_rows(split_dir, selected_ids, output_csv)

    with open(sample_id_dir / f"{split}_sample_ids.txt", "w", encoding="utf-8") as handle:
        for sample_id in sorted(selected_ids):
            handle.write(sample_id + "\n")

    write_metadata_csv(report_dir / f"{split}_selected_metadata.csv", selected)
    split_report = write_report(
        report_dir / f"{split}_report.json",
        split=split,
        target=target,
        selected=selected,
        scan_stats=scan_stats,
        selection_report=selection_report,
        exported_rows=exported_rows,
    )
    print(
        f"[build] {split}: selected={len(selected)} exported_rows={exported_rows} "
        f"unique_mmsi={split_report['unique_mmsi']} unique_segments={split_report['unique_segments']}"
    )
    return split_report


def main() -> None:
    args = parse_args()
    allowed_groups = {
        item.strip() for item in args.allow_ship_groups.split(",") if item.strip()
    }
    excluded_groups = {
        item.strip() for item in args.exclude_ship_groups.split(",") if item.strip()
    }

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": {
            "benchmark_root": str(args.benchmark_root),
            "output_root": str(args.output_root),
            "train_target": args.train_target,
            "val_target": args.val_target,
            "test_target": args.test_target,
            "allow_ship_groups": sorted(allowed_groups),
            "exclude_ship_groups": sorted(excluded_groups),
            "max_per_mmsi": args.max_per_mmsi,
            "max_per_segment": args.max_per_segment,
            "min_segment_step_gap": args.min_segment_step_gap,
            "min_quality_score": args.min_quality_score,
        }
    }

    for split, target in (
        ("train", args.train_target),
        ("val", args.val_target),
        ("test", args.test_target),
    ):
        print(f"[build] split={split} target={target}")
        summary[split] = build_split(
            split=split,
            target=target,
            benchmark_root=args.benchmark_root,
            output_root=args.output_root,
            args=args,
            allowed_groups=allowed_groups,
            excluded_groups=excluded_groups,
        )

    with open(args.output_root / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print("[build] finished")


if __name__ == "__main__":
    main()
