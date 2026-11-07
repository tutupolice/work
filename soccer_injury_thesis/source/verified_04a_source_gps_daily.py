#!/usr/bin/env python3
"""SoccerMon source-domain GPS audit and daily aggregation (locked v1.4).

This script replaces the legacy ``phase1_add_gps_data.py`` implementation.
The legacy implementation summed every speed row under a fixed 0.1-second
assumption. SoccerMon parquet files contain high-rate inertial rows and roughly
ten duplicate rows per 0.1-second GPS timestamp, so that approach inflates
distance substantially.

Locked processing order
-----------------------
1. Read only columns needed for speed/heart-rate aggregation.
2. Collapse equal timestamps; speed and heart rate use the median.
3. Keep speeds in the conservative physical QC interval [0, 10] m/s.
4. Integrate speed over observed time with the trapezoidal rule.
5. Do not bridge a gap greater than max(1 s, 2 x median positive interval).
6. Audit gross spatial extent and exact 11.5 m/s saturation plateaus. A file
   is excluded from load aggregation only when an exact saturation plateau
   lasts at least 1 second. A robust 1st--99th percentile spatial diagonal
   above 1 km is retained as a coordinate-quality warning; it does not by
   itself invalidate otherwise plausible speed-derived load. The raw audit
   row is retained in both cases.
7. Calculate distance above several speed thresholds. The primary HSRD
   definition is >4.44 m/s (15.984 km/h), matching the SoccerMon technical
   validation; 12.5, 15.0, and 19.98 km/h are sensitivity definitions.

No outcome or injury file is read by this script.

Examples
--------
Smoke test (does not overwrite the final cache)::

    python verified_04a_source_gps_daily.py --limit 20 \
        --output-dir verified_outputs/04_source_gps_smoke --no-resume

Full resumable run::

    python verified_04a_source_gps_daily.py --workers 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCHEMA_VERSION = "1.4.0"
UPGRADABLE_SCHEMA_VERSION = "1.3.0"
ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "verified_outputs" / "04_source_gps"
GPS_ROOTS = (ROOT / "data" / "2020", ROOT / "data" / "2021")
SPEED_MIN_M_S = 0.0
SPEED_MAX_M_S = 10.0
# Raw STATSports exports in this local copy contain long, exact plateaus at
# 11.5 m/s (41.4 km/h), including travel across large geographic distances.
# This is treated as a device saturation/non-soccer artifact, not athlete
# locomotion. Exact/near-exact 11.5 values are removed before integration.
SATURATED_SPEED_M_S = 11.5
SATURATED_SPEED_ATOL = 1e-9
SATURATION_PLATEAU_MIN_DURATION_S = 1.0
ROBUST_SPATIAL_QUANTILES = (0.01, 0.99)
# A regulation pitch is at most 120 x 90 m (diagonal 150 m). One kilometre is
# deliberately generous and is used to flag gross coordinate movement. It is
# not an automatic speed-load exclusion because a spatial-only audit found
# plausible speed-derived distance/max-speed values in affected sessions.
MAX_ROBUST_SPATIAL_EXTENT_M = 1000.0
PRIMARY_HIR_M_S = 4.44
PRIMARY_HIR_KM_H = PRIMARY_HIR_M_S * 3.6
HIR_THRESHOLDS_KM_H = (12.5, 15.0, PRIMARY_HIR_KM_H, 19.98)
CHECKPOINT_NAME = "source_gps_file_checkpoint.jsonl"
PUBLISHED_GPS_REPORT_COUNTS = {"TeamA": 5438, "TeamB": 4637, "total": 10075}
SOURCE_PAPER = "https://doi.org/10.1038/s41597-024-03386-x"
IFAB_FIELD_DIMENSIONS = "https://www.theifab.com/laws/latest/the-field-of-play/"


def native(value: Any) -> Any:
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    return value


def threshold_key(threshold_km_h: float) -> str:
    """Stable column-name token for a speed threshold."""
    if math.isclose(threshold_km_h, round(threshold_km_h), abs_tol=1e-12):
        token = f"{threshold_km_h:.1f}"
    else:
        token = f"{threshold_km_h:.3f}".rstrip("0")
    return token.replace(".", "p")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_identity(path: Path) -> tuple[str, str]:
    """Return ISO date and anonymized athlete ID from a parquet filename."""
    stem = path.stem
    match = re.match(r"^(\d{4}-\d{2}-\d{2})-(Team[AB]-.+)$", stem)
    if not match:
        raise ValueError(f"unexpected parquet filename: {path.name}")
    return match.group(1), match.group(2)


def seconds_since_midnight(values: pd.Series) -> pd.Series:
    parsed = pd.to_timedelta(values.astype("string"), errors="coerce")
    return parsed.dt.total_seconds()


def longest_plateau_duration_s(
    point_mask: np.ndarray,
    seconds: np.ndarray,
    gap_cutoff_s: float,
) -> float:
    """Return the longest continuously sampled run for a point-level mask."""
    if point_mask.size < 2:
        return 0.0
    dt = np.diff(seconds)
    best = 0.0
    current = 0.0
    for index, interval in enumerate(dt):
        if (
            point_mask[index]
            and point_mask[index + 1]
            and np.isfinite(interval)
            and 0 < interval <= gap_cutoff_s
        ):
            current += float(interval)
            best = max(best, current)
        else:
            current = 0.0
    return best


def robust_spatial_extent_m(latitude: np.ndarray, longitude: np.ndarray) -> tuple[int, float]:
    """Estimate a robust geographic bounding-box diagonal in metres.

    The 1st--99th percentile box ignores isolated coordinate glitches while
    retaining sustained travel. Zero/zero coordinates are treated as missing.
    """
    valid = (
        np.isfinite(latitude)
        & np.isfinite(longitude)
        & (np.abs(latitude) <= 90)
        & (np.abs(longitude) <= 180)
        & ~((latitude == 0) & (longitude == 0))
    )
    count = int(valid.sum())
    if count < 2:
        return count, np.nan
    low_q, high_q = ROBUST_SPATIAL_QUANTILES
    lat_low, lat_high = np.quantile(latitude[valid], [low_q, high_q])
    lon_low, lon_high = np.quantile(longitude[valid], [low_q, high_q])
    latitude_mid = float(np.median(latitude[valid]))
    north_south_m = abs(float(lat_high - lat_low)) * 111_320.0
    east_west_m = (
        abs(float(lon_high - lon_low))
        * 111_320.0
        * abs(math.cos(math.radians(latitude_mid)))
    )
    return count, float(math.hypot(north_south_m, east_west_m))


def integrate_above_threshold(
    v0: np.ndarray,
    v1: np.ndarray,
    dt: np.ndarray,
    segment_ok: np.ndarray,
    threshold_m_s: float,
) -> float:
    """Integrate the part of linearly interpolated segments above threshold."""
    out = np.zeros_like(dt, dtype=float)
    # SoccerMon reports HSRD using a strict ``speed > threshold`` rule. A
    # segment flat exactly on the cut-point therefore contributes nothing.
    # Crossing end-points themselves have zero duration, so interpolation is
    # unchanged for segments that enter or leave the high-speed zone.
    both = segment_ok & (v0 > threshold_m_s) & (v1 > threshold_m_s)
    out[both] = 0.5 * (v0[both] + v1[both]) * dt[both]

    up = segment_ok & (v0 <= threshold_m_s) & (v1 > threshold_m_s) & (v1 > v0)
    if np.any(up):
        crossing = (threshold_m_s - v0[up]) / (v1[up] - v0[up])
        duration = dt[up] * (1.0 - crossing)
        out[up] = 0.5 * (threshold_m_s + v1[up]) * duration

    down = segment_ok & (v0 > threshold_m_s) & (v1 <= threshold_m_s) & (v0 > v1)
    if np.any(down):
        crossing = (v0[down] - threshold_m_s) / (v0[down] - v1[down])
        duration = dt[down] * crossing
        out[down] = 0.5 * (v0[down] + threshold_m_s) * duration
    return float(np.nansum(out))


def process_file(path_string: str) -> dict[str, Any]:
    path = Path(path_string)
    started = time.perf_counter()
    base: dict[str, Any] = {
        "qc_schema_version": SCHEMA_VERSION,
        "relative_path": str(path.relative_to(ROOT)),
        "file_size_bytes": path.stat().st_size,
        "status": "error",
        "error_type": "",
        "error_message": "",
    }
    try:
        date, file_player = parse_identity(path)
        frame = pd.read_parquet(
            path,
            columns=["player_name", "time", "lat", "lon", "speed", "heart_rate"],
        )
        base.update({"date": date, "player_name": file_player, "raw_row_count": len(frame)})
        if frame.empty:
            raise ValueError("empty parquet file")

        observed_players = frame["player_name"].dropna().astype(str).unique().tolist()
        base["embedded_player_count"] = len(observed_players)
        base["filename_player_matches"] = int(observed_players == [file_player])

        compact = pd.DataFrame(
            {
                "second": seconds_since_midnight(frame["time"]),
                "latitude": pd.to_numeric(frame["lat"], errors="coerce"),
                "longitude": pd.to_numeric(frame["lon"], errors="coerce"),
                "speed": pd.to_numeric(frame["speed"], errors="coerce"),
                "heart_rate": pd.to_numeric(frame["heart_rate"], errors="coerce"),
            }
        ).dropna(subset=["second"])

        grouped = (
            compact.groupby("second", as_index=False, sort=True)
            .agg(
                latitude=("latitude", "median"),
                longitude=("longitude", "median"),
                speed=("speed", "median"),
                heart_rate=("heart_rate", "median"),
            )
            .sort_values("second")
            .reset_index(drop=True)
        )
        n_unique = len(grouped)
        if n_unique < 2:
            raise ValueError("fewer than two valid unique timestamps")

        seconds = grouped["second"].to_numpy(dtype=float)
        speed = grouped["speed"].to_numpy(dtype=float)
        positive_dt = np.diff(seconds)
        positive_dt = positive_dt[np.isfinite(positive_dt) & (positive_dt > 0)]
        if positive_dt.size == 0:
            raise ValueError("no positive timestamp interval")
        median_dt = float(np.median(positive_dt))
        gap_cutoff = max(1.0, 2.0 * median_dt)

        v0 = speed[:-1]
        v1 = speed[1:]
        dt = np.diff(seconds)
        saturated_speed = np.isclose(
            speed, SATURATED_SPEED_M_S, rtol=0.0, atol=SATURATED_SPEED_ATOL
        )
        saturation_plateau_s = longest_plateau_duration_s(
            saturated_speed, seconds, gap_cutoff
        )
        geo_count, spatial_extent_m = robust_spatial_extent_m(
            grouped["latitude"].to_numpy(dtype=float),
            grouped["longitude"].to_numpy(dtype=float),
        )
        exclusion_reasons: list[str] = []
        warning_reasons: list[str] = []
        if saturation_plateau_s >= SATURATION_PLATEAU_MIN_DURATION_S:
            exclusion_reasons.append("exact_11p5_m_s_plateau_ge_1s")
        if np.isfinite(spatial_extent_m) and spatial_extent_m > MAX_ROBUST_SPATIAL_EXTENT_M:
            warning_reasons.append("robust_spatial_extent_gt_1km")
        file_qc_excluded = bool(exclusion_reasons)

        valid_point = (
            np.isfinite(speed)
            & (speed >= SPEED_MIN_M_S)
            & (speed <= SPEED_MAX_M_S)
            & ~saturated_speed
        )
        segment_ok = (
            np.isfinite(dt)
            & (dt > 0)
            & (dt <= gap_cutoff)
            & valid_point[:-1]
            & valid_point[1:]
        )
        segment_distance = np.zeros_like(dt, dtype=float)
        segment_distance[segment_ok] = 0.5 * (v0[segment_ok] + v1[segment_ok]) * dt[segment_ok]
        pre_qc_valid_duration = float(np.sum(dt[segment_ok]))
        pre_qc_total_distance = float(np.sum(segment_distance))
        pre_qc_usable = bool(segment_ok.any() and pre_qc_valid_duration > 0)
        aggregation_usable = pre_qc_usable and not file_qc_excluded
        valid_duration = pre_qc_valid_duration if aggregation_usable else np.nan
        total_distance = pre_qc_total_distance if aggregation_usable else np.nan

        hr = grouped["heart_rate"].to_numpy(dtype=float)
        valid_hr = np.isfinite(hr) & (hr >= 30) & (hr <= 240)

        result = {
            **base,
            "status": "ok",
            "unique_timestamp_count": n_unique,
            "duplicate_row_multiplier": float(len(frame) / n_unique),
            "median_sampling_interval_s": median_dt,
            "gap_cutoff_s": gap_cutoff,
            "span_duration_s": float(seconds[-1] - seconds[0]),
            "pre_qc_valid_integrated_duration_s": pre_qc_valid_duration,
            "valid_integrated_duration_s": valid_duration,
            "excluded_gap_duration_s": float(np.sum(dt[(dt > gap_cutoff) & np.isfinite(dt)])),
            "max_timestamp_gap_s": float(np.nanmax(dt)),
            "valid_coordinate_timestamp_count": geo_count,
            "robust_spatial_extent_m": spatial_extent_m,
            "saturation_longest_plateau_s": saturation_plateau_s,
            "file_qc_excluded": int(file_qc_excluded),
            "file_qc_exclusion_reason": "|".join(exclusion_reasons),
            "file_qc_warning": int(bool(warning_reasons)),
            "file_qc_warning_reason": "|".join(warning_reasons),
            "valid_speed_point_count": int(valid_point.sum()),
            "invalid_speed_point_count": int((~valid_point).sum()),
            "invalid_speed_fraction": float((~valid_point).mean()),
            "saturated_speed_point_count": int(saturated_speed.sum()),
            "saturated_speed_fraction": float(saturated_speed.mean()),
            "pre_qc_total_distance_m": pre_qc_total_distance,
            "total_distance_m": total_distance,
            "average_speed_time_weighted_m_s": (
                total_distance / valid_duration
                if aggregation_usable and valid_duration > 0
                else np.nan
            ),
            "pre_qc_maximum_speed_m_s": (
                float(np.nanmax(speed[valid_point])) if valid_point.any() else np.nan
            ),
            "maximum_speed_m_s": (
                float(np.nanmax(speed[valid_point]))
                if aggregation_usable and valid_point.any()
                else np.nan
            ),
            "heart_rate_mean_bpm": (
                float(np.nanmean(hr[valid_hr]))
                if aggregation_usable and valid_hr.any()
                else np.nan
            ),
            "heart_rate_max_bpm": (
                float(np.nanmax(hr[valid_hr]))
                if aggregation_usable and valid_hr.any()
                else np.nan
            ),
            "heart_rate_valid_fraction": float(valid_hr.mean()),
            "gps_usable": int(aggregation_usable),
        }
        for threshold in HIR_THRESHOLDS_KM_H:
            key = threshold_key(threshold)
            pre_qc_threshold_distance = integrate_above_threshold(
                v0, v1, dt, segment_ok, threshold / 3.6
            )
            result[f"pre_qc_distance_above_{key}_km_h_m"] = pre_qc_threshold_distance
            result[f"distance_above_{key}_km_h_m"] = (
                pre_qc_threshold_distance if aggregation_usable else np.nan
            )
        result["processing_seconds"] = time.perf_counter() - started
        return native(result)
    except Exception as exc:  # audit errors rather than silently dropping files
        base.update(
            {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "processing_seconds": time.perf_counter() - started,
            }
        )
        return native(base)


def discover_files() -> list[Path]:
    files: list[Path] = []
    for root in GPS_ROOTS:
        files.extend(root.rglob("*.parquet"))
    return sorted(files)


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            old_max = record.get("maximum_speed_m_s")
            contains_unfiltered_saturation = (
                old_max is not None
                and np.isfinite(old_max)
                and np.isclose(
                    old_max,
                    SATURATED_SPEED_M_S,
                    rtol=0.0,
                    atol=SATURATED_SPEED_ATOL,
                )
                and "saturated_speed_fraction" not in record
            )
            record_schema = record.get("qc_schema_version")
            if record.get("status") == "ok" and record_schema == UPGRADABLE_SCHEMA_VERSION:
                saturation_excluded = (
                    float(record.get("saturation_longest_plateau_s") or 0.0)
                    >= SATURATION_PLATEAU_MIN_DURATION_S
                )
                spatial_warning = (
                    record.get("robust_spatial_extent_m") is not None
                    and np.isfinite(record["robust_spatial_extent_m"])
                    and float(record["robust_spatial_extent_m"])
                    > MAX_ROBUST_SPATIAL_EXTENT_M
                )
                # Spatial-only files were excluded under v1.3, so their HR and
                # aggregate fields must be recovered from raw data. All other
                # records can be upgraded deterministically without rereading.
                if spatial_warning and not saturation_excluded:
                    continue
                record["qc_schema_version"] = SCHEMA_VERSION
                record["file_qc_excluded"] = int(saturation_excluded)
                record["file_qc_exclusion_reason"] = (
                    "exact_11p5_m_s_plateau_ge_1s" if saturation_excluded else ""
                )
                record["file_qc_warning"] = int(spatial_warning)
                record["file_qc_warning_reason"] = (
                    "robust_spatial_extent_gt_1km" if spatial_warning else ""
                )
            if (
                record.get("status") == "ok"
                and record.get("qc_schema_version") == SCHEMA_VERSION
                and not contains_unfiltered_saturation
            ):
                records[record["relative_path"]] = record
    return records


def aggregate_daily(file_audit: pd.DataFrame) -> pd.DataFrame:
    ok = file_audit.loc[file_audit["status"].eq("ok")].copy()
    if ok.empty:
        return pd.DataFrame()
    ok["weighted_speed_numerator"] = (
        ok["average_speed_time_weighted_m_s"] * ok["valid_integrated_duration_s"]
    )
    ok["weighted_hr_numerator"] = ok["heart_rate_mean_bpm"] * ok["valid_integrated_duration_s"]
    distance_cols = ["total_distance_m"] + [
        f"distance_above_{threshold_key(v)}_km_h_m" for v in HIR_THRESHOLDS_KM_H
    ]
    aggregations: dict[str, Any] = {
        "gps_file_count": ("relative_path", "size"),
        "gps_usable_file_count": ("gps_usable", "sum"),
        "gps_qc_excluded_file_count": ("file_qc_excluded", "sum"),
        "gps_qc_warning_file_count": ("file_qc_warning", "sum"),
        "gps_raw_row_count": ("raw_row_count", "sum"),
        "gps_unique_timestamp_count": ("unique_timestamp_count", "sum"),
        "gps_valid_duration_s": (
            "valid_integrated_duration_s",
            lambda values: values.sum(min_count=1),
        ),
        "gps_max_speed_m_s": ("maximum_speed_m_s", "max"),
        "gps_invalid_speed_point_count": ("invalid_speed_point_count", "sum"),
        "gps_saturated_speed_point_count": ("saturated_speed_point_count", "sum"),
        "gps_valid_speed_point_count": ("valid_speed_point_count", "sum"),
        "gps_weighted_speed_numerator": (
            "weighted_speed_numerator",
            lambda values: values.sum(min_count=1),
        ),
        "gps_weighted_hr_numerator": (
            "weighted_hr_numerator",
            lambda values: values.sum(min_count=1),
        ),
        "gps_usable": ("gps_usable", "max"),
    }
    for column in distance_cols:
        aggregations[column] = (column, lambda values: values.sum(min_count=1))
    daily = ok.groupby(["date", "player_name"], as_index=False).agg(**aggregations)
    daily["gps_average_speed_time_weighted_m_s"] = (
        daily["gps_weighted_speed_numerator"] / daily["gps_valid_duration_s"].replace(0, np.nan)
    )
    daily["gps_heart_rate_mean_bpm"] = (
        daily["gps_weighted_hr_numerator"] / daily["gps_valid_duration_s"].replace(0, np.nan)
    )
    daily["gps_invalid_speed_fraction"] = daily["gps_invalid_speed_point_count"] / (
        daily["gps_invalid_speed_point_count"] + daily["gps_valid_speed_point_count"]
    ).replace(0, np.nan)
    daily["gps_saturated_speed_fraction"] = daily["gps_saturated_speed_point_count"] / (
        daily["gps_invalid_speed_point_count"] + daily["gps_valid_speed_point_count"]
    ).replace(0, np.nan)
    daily = daily.drop(columns=["gps_weighted_speed_numerator", "gps_weighted_hr_numerator"])
    return daily.sort_values(["player_name", "date"]).reset_index(drop=True)


def write_report(output_dir: Path, manifest: dict[str, Any]) -> None:
    summary = manifest["summary"]
    lines = [
        "# SoccerMon GPS 日级聚合审计",
        "",
        f"- schema_version: `{SCHEMA_VERSION}`",
        f"- 发现 parquet 文件: **{summary['discovered_file_count']:,}**",
        f"- SoccerMon论文报告GPS会话: **{summary['published_report_count']:,}**",
        f"- 本机清单与论文总数差: **{summary['local_minus_published_report_count']:+,}**",
        f"- 成功处理: **{summary['successful_file_count']:,}**",
        f"- 失败: **{summary['failed_file_count']:,}**",
        f"- 清单中无审计记录: **{summary['missing_checkpoint_record_count']:,}**",
        f"- 日级运动员记录: **{summary['daily_player_date_count']:,}**",
        f"- GPS 运动员: **{summary['gps_player_count']:,}**",
        f"- 主 HSRD 阈值: **>{PRIMARY_HIR_M_S:.2f} m/s ({PRIMARY_HIR_KM_H:.3f} km/h)**",
        f"- 含精确11.5 m/s速度值的文件: **{summary['files_with_saturated_speed']:,}**",
        f"- 因持续饱和平台排除的文件: **{summary['files_excluded_saturation_plateau']:,}**",
        f"- 稳健空间跨度>1 km警告文件: **{summary['files_flagged_spatial_extent']:,}**",
        f"- 任一文件级QC排除: **{summary['files_excluded_by_qc']:,}**",
        "",
        "## 锁定计算规则",
        "",
        "相同时间戳先取速度、心率及经纬度中位数；速度限定为0–10 m/s，精确11.5 m/s值不参与积分；距离按实际时间间隔做梯形积分。",
        "超过max(1 s, 2×会话中位采样间隔)的间断不连线。若精确11.5 m/s连续平台达到1 s，则整份文件仅保留审计、不进入负荷聚合。",
        f"主要HSRD采用SoccerMon技术验证定义>{PRIMARY_HIR_M_S:.2f} m/s；12.5、15.0和19.98 km/h仅作敏感性指标。",
        "经纬度1%–99%稳健包围框对角线超过1 km仅记坐标质量警告，不单独排除速度负荷；17份空间单异常文件的专项复核显示其速度积分距离和最大速度大多仍处合理训练范围。",
        "旧脚本按原始行直接求和的结果不得用于论文。",
        "",
    ]
    (output_dir / "source_gps_aggregation_audit.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / CHECKPOINT_NAME
    if args.no_resume and checkpoint.exists():
        checkpoint.unlink()

    files = discover_files()
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise SystemExit("No parquet files discovered.")

    inventory_text = "\n".join(
        f"{p.relative_to(ROOT)}\t{p.stat().st_size}\t{p.stat().st_mtime_ns}" for p in files
    )
    inventory_digest = hashlib.sha256(inventory_text.encode("utf-8")).hexdigest()
    team_file_counts = {
        "TeamA": sum("-TeamA-" in path.name for path in files),
        "TeamB": sum("-TeamB-" in path.name for path in files),
    }
    cached = {} if args.no_resume else load_checkpoint(checkpoint)
    pending = [p for p in files if str(p.relative_to(ROOT)) not in cached]

    print("=" * 76)
    print(f"SoccerMon GPS 日级审计聚合（锁定版 v{SCHEMA_VERSION}）")
    print("=" * 76)
    print(f"发现文件: {len(files):,}; 已缓存: {len(cached):,}; 待处理: {len(pending):,}")
    print(f"workers={args.workers}; 输出={output_dir}")

    completed = 0
    completed_relative_paths: set[str] = set()
    started = time.perf_counter()
    mode = "a" if checkpoint.exists() and not args.no_resume else "w"
    with checkpoint.open(mode, encoding="utf-8") as handle:
        if args.workers <= 1:
            iterator = (process_file(str(path)) for path in pending)
            for record in iterator:
                handle.write(json.dumps(native(record), ensure_ascii=False) + "\n")
                handle.flush()
                completed += 1
                completed_relative_paths.add(record["relative_path"])
                if completed % 25 == 0 or completed == len(pending):
                    print(f"  进度 {completed:,}/{len(pending):,}", flush=True)
        else:
            try:
                with ProcessPoolExecutor(max_workers=args.workers) as pool:
                    futures = {pool.submit(process_file, str(path)): path for path in pending}
                    for future in as_completed(futures):
                        record = future.result()
                        handle.write(json.dumps(native(record), ensure_ascii=False) + "\n")
                        handle.flush()
                        completed += 1
                        completed_relative_paths.add(record["relative_path"])
                        if completed % 25 == 0 or completed == len(pending):
                            elapsed = time.perf_counter() - started
                            rate = completed / elapsed if elapsed else 0
                            eta = (len(pending) - completed) / rate if rate else math.nan
                            print(
                                f"  进度 {completed:,}/{len(pending):,}; "
                                f"{rate:.2f} 文件/s; ETA {eta/60:.1f} min",
                                flush=True,
                            )
            except (PermissionError, OSError) as exc:
                # Some managed/sandboxed macOS environments prohibit process
                # semaphores. Preserve reproducibility by falling back to the
                # same deterministic worker function sequentially.
                print(f"  多进程不可用 ({type(exc).__name__}: {exc})，自动回退线程池。")
                remaining = [
                    path
                    for path in pending
                    if str(path.relative_to(ROOT)) not in completed_relative_paths
                ]
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    futures = {pool.submit(process_file, str(path)): path for path in remaining}
                    for future in as_completed(futures):
                        record = future.result()
                        handle.write(json.dumps(native(record), ensure_ascii=False) + "\n")
                        handle.flush()
                        completed += 1
                        completed_relative_paths.add(record["relative_path"])
                        if completed % 25 == 0 or completed == len(pending):
                            elapsed = time.perf_counter() - started
                            rate = completed / elapsed if elapsed else 0
                            eta = (len(pending) - completed) / rate if rate else math.nan
                            print(
                                f"  进度 {completed:,}/{len(pending):,}; "
                                f"{rate:.2f} 文件/s; ETA {eta/60:.1f} min",
                                flush=True,
                            )

    records: list[dict[str, Any]] = []
    with checkpoint.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        latest[record["relative_path"]] = record
    selected = [latest[str(path.relative_to(ROOT))] for path in files if str(path.relative_to(ROOT)) in latest]
    for record in selected:
        record.setdefault("saturated_speed_point_count", 0)
        record.setdefault("saturated_speed_fraction", 0.0)
        if record.get("status") == "ok":
            saturation_excluded = (
                float(record.get("saturation_longest_plateau_s") or 0.0)
                >= SATURATION_PLATEAU_MIN_DURATION_S
            )
            spatial_warning = (
                record.get("robust_spatial_extent_m") is not None
                and np.isfinite(record["robust_spatial_extent_m"])
                and float(record["robust_spatial_extent_m"])
                > MAX_ROBUST_SPATIAL_EXTENT_M
            )
            if (
                record.get("qc_schema_version") == UPGRADABLE_SCHEMA_VERSION
                and spatial_warning
                and not saturation_excluded
            ):
                raise RuntimeError(
                    "spatial-only v1.3 record was not reprocessed: "
                    f"{record['relative_path']}"
                )
            record["qc_schema_version"] = SCHEMA_VERSION
            record["file_qc_excluded"] = int(saturation_excluded)
            record["file_qc_exclusion_reason"] = (
                "exact_11p5_m_s_plateau_ge_1s" if saturation_excluded else ""
            )
            record["file_qc_warning"] = int(spatial_warning)
            record["file_qc_warning_reason"] = (
                "robust_spatial_extent_gt_1km" if spatial_warning else ""
            )
    missing_checkpoint_records = len(files) - len(selected)
    # Rewrite a canonical one-record-per-file checkpoint. This removes stale
    # retry rows and converts non-finite values to JSON null, yielding strict
    # JSONL that can be independently audited by non-Python tools.
    canonical_checkpoint = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
    with canonical_checkpoint.open("w", encoding="utf-8") as handle:
        for record in selected:
            handle.write(json.dumps(native(record), ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(canonical_checkpoint, checkpoint)
    audit = pd.DataFrame(selected).sort_values("relative_path").reset_index(drop=True)
    daily = aggregate_daily(audit)

    audit.to_csv(output_dir / "source_gps_file_audit.csv", index=False)
    daily.to_csv(output_dir / "source_gps_daily.csv", index=False)

    success = int(audit["status"].eq("ok").sum())
    failure = int(audit["status"].ne("ok").sum())
    gps_complete = failure == 0 and missing_checkpoint_records == 0 and success == len(files)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "lock_status": "PASSED" if gps_complete else "FAILED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "input_inventory_sha256": inventory_digest,
        "parameters": {
            "speed_qc_m_s": [SPEED_MIN_M_S, SPEED_MAX_M_S],
            "saturated_speed_artifact_m_s": SATURATED_SPEED_M_S,
            "saturation_plateau_exclusion_s": SATURATION_PLATEAU_MIN_DURATION_S,
            "robust_spatial_quantiles": list(ROBUST_SPATIAL_QUANTILES),
            "maximum_robust_spatial_extent_warning_m": MAX_ROBUST_SPATIAL_EXTENT_M,
            "spatial_extent_policy": "quality warning only; not an automatic speed-load exclusion",
            "hir_thresholds_km_h": list(HIR_THRESHOLDS_KM_H),
            "primary_hir_threshold_km_h": PRIMARY_HIR_KM_H,
            "primary_hir_threshold_m_s": PRIMARY_HIR_M_S,
            "gap_rule": "dt <= max(1 second, 2 x session median positive dt)",
            "timestamp_collapse": "median speed and heart rate",
            "distance_integration": "linear/trapezoidal",
        },
        "summary": {
            "discovered_file_count": len(files),
            "discovered_team_a_file_count": team_file_counts["TeamA"],
            "discovered_team_b_file_count": team_file_counts["TeamB"],
            "published_report_count": PUBLISHED_GPS_REPORT_COUNTS["total"],
            "published_team_a_report_count": PUBLISHED_GPS_REPORT_COUNTS["TeamA"],
            "published_team_b_report_count": PUBLISHED_GPS_REPORT_COUNTS["TeamB"],
            "local_minus_published_report_count": len(files) - PUBLISHED_GPS_REPORT_COUNTS["total"],
            "successful_file_count": success,
            "failed_file_count": failure,
            "missing_checkpoint_record_count": missing_checkpoint_records,
            "files_with_saturated_speed": int(audit["saturated_speed_point_count"].gt(0).sum()),
            "total_saturated_speed_points": int(audit["saturated_speed_point_count"].sum()),
            "files_excluded_saturation_plateau": int(
                audit["saturation_longest_plateau_s"]
                .ge(SATURATION_PLATEAU_MIN_DURATION_S)
                .sum()
            ),
            "files_flagged_spatial_extent": int(
                audit["robust_spatial_extent_m"].gt(MAX_ROBUST_SPATIAL_EXTENT_M).sum()
            ),
            "files_excluded_by_qc": int(audit["file_qc_excluded"].eq(1).sum()),
            "daily_player_date_count": len(daily),
            "daily_player_date_with_usable_gps_count": int(daily["gps_usable"].eq(1).sum()),
            "gps_player_count": int(daily["player_name"].nunique()) if not daily.empty else 0,
        },
        "outputs": {
            "file_audit": "source_gps_file_audit.csv",
            "daily_features": "source_gps_daily.csv",
            "report": "source_gps_aggregation_audit.md",
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "references": {
            "soccermon_data_paper_and_hsrd_definition": SOURCE_PAPER,
            "ifab_field_dimensions": IFAB_FIELD_DIMENSIONS,
        },
    }
    with (output_dir / "source_gps_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(native(manifest), handle, ensure_ascii=False, indent=2)
    write_report(output_dir, manifest)

    print("-" * 76)
    print(f"成功 {success:,}; 失败 {failure:,}; 日级记录 {len(daily):,}")
    if not gps_complete:
        print(
            "GPS审计不完整：请检查失败文件和缺失checkpoint记录，主流程不得静默继续。"
        )
    else:
        print("GPS LOCK STATUS: PASSED")


if __name__ == "__main__":
    main()
