#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SoccerMon source-domain preprocessing and athlete-day data integration.

This script is intentionally limited to preprocessing/alignment.  It:

* reads the real PmSys subjective sources and the verified daily GPS cache;
* converts session-level training records and wide daily exports to one
  ``player_name + date`` key;
* applies no imputation, standardization, window construction, feature
  selection, or model fitting;
* writes a complete source-key union for audit and a separate eligible
  prediction-origin table for the subsequent feature stage.

The locked injury report-day table is carried only as outcome/QC metadata.
Injury location and severity are never read into the panel.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SUBJECTIVE = ROOT / "data/subjective"
GPS_DAILY = ROOT / "verified_outputs/04_source_gps/source_gps_daily.csv"
INJURY_REPORT_DAY = ROOT / (
    "verified_outputs/07_source_injury_reportday_t1_v1/04_injury_report_day.csv"
)
OUTPUT_DIR = ROOT / "verified_outputs/01_source_preprocessing_v1"

TRAINING_EXPORTS = {
    "daily_load": "daily_load_au",
    "weekly_load": "weekly_load_au",
    "atl": "atl_exported_au",
    "ctl28": "ctl28_exported_au",
    "ctl42": "ctl42_exported_au",
    "acwr": "acwr_exported",
    "monotony": "monotony_exported",
    "strain": "strain_exported_au",
}

WELLNESS_RAW = {
    "fatigue": "fatigue_raw",
    "mood": "mood_raw",
    "readiness": "readiness_raw",
    "sleep_duration": "sleep_duration_raw",
    "sleep_quality": "sleep_quality_raw",
    "soreness": "soreness_raw",
    "stress": "stress_raw",
}

DO_NOT_USE_AS_PREDICTOR = {
    "injury_report_t",
    "target_date",
    "y_t1",
    "outcome_boundary_flag",
    "monitoring_last_date",
    "pmsys_eligible",
    "calendar_year",
    "origin_eligible",
    "origin_eligibility_reason",
    "player_name",
    "date",
    "team",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def native(value: Any) -> Any:
    """Convert numpy/pandas/path objects to JSON-safe values."""
    if isinstance(value, dict):
        return {str(key): native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [native(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def parse_wide_ddmmyyyy(path: Path, value_name: str) -> pd.DataFrame:
    """Read a PmSys wide CSV whose first column is DD.MM.YYYY."""
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Empty source file: {path}")
    date_column = frame.columns[0]
    frame[date_column] = pd.to_datetime(
        frame[date_column], format="%d.%m.%Y", errors="coerce"
    ).dt.normalize()
    if frame[date_column].isna().any():
        raise ValueError(f"Invalid DD.MM.YYYY date in {path}")

    long = frame.melt(
        id_vars=[date_column], var_name="player_name", value_name=value_name
    ).rename(columns={date_column: "date"})
    long["player_name"] = long["player_name"].astype(str).str.strip()
    long[value_name] = pd.to_numeric(long[value_name], errors="coerce")
    if long.duplicated(["player_name", "date"]).any():
        raise AssertionError(f"Duplicate player_name + date keys in {path}")
    return long[["player_name", "date", value_name]]


def load_sessions(path: Path) -> pd.DataFrame:
    """Aggregate session.json to player-day without imputing missing values."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for player_name, sessions in raw.items():
        for source_row, session in enumerate(sessions, start=1):
            rows.append(
                {
                    "player_name": str(player_name).strip(),
                    "date": pd.to_datetime(
                        session.get("date"), format="%d.%m.%Y", errors="raise"
                    ).normalize(),
                    "session_source_row": source_row,
                    "session_srpe_au": pd.to_numeric(
                        session.get("srpe"), errors="coerce"
                    ),
                    "session_rpe": pd.to_numeric(
                        session.get("rpe"), errors="coerce"
                    ),
                    "session_duration_min": pd.to_numeric(
                        session.get("duration"), errors="coerce"
                    ),
                }
            )
    sessions = pd.DataFrame(rows)
    if sessions.empty:
        raise ValueError("session.json contains no records")

    grouped = sessions.groupby(["player_name", "date"], sort=False)
    daily = grouped.agg(
        session_count=("session_source_row", "size"),
        session_rpe_mean=("session_rpe", "mean"),
    ).reset_index()
    daily["session_srpe_sum_au"] = grouped["session_srpe_au"].sum(
        min_count=1
    ).to_numpy()
    daily["session_duration_sum_min"] = grouped["session_duration_min"].sum(
        min_count=1
    ).to_numpy()
    daily["has_training_record"] = np.int8(1)
    return daily


def load_wellness() -> pd.DataFrame:
    """Load all seven wellness files and derive the verified burden fields."""
    merged: pd.DataFrame | None = None
    raw_fields: list[str] = []
    for source_name, field_name in WELLNESS_RAW.items():
        path = SUBJECTIVE / "wellness" / f"{source_name}.csv"
        current = parse_wide_ddmmyyyy(path, field_name)
        raw_fields.append(field_name)
        merged = (
            current
            if merged is None
            else merged.merge(
                current,
                on=["player_name", "date"],
                how="outer",
                validate="one_to_one",
            )
        )

    if merged is None:
        raise ValueError("No wellness files were loaded")
    merged["has_wellness_record"] = (
        merged[raw_fields].notna().any(axis=1).astype("int8")
    )

    # PmSys wellness burden harmonization: raw 1 = worst and raw 5 = best.
    # This is a deterministic source-scale transformation, not imputation.
    burden_mapping = {
        "fatigue_raw": "fatigue_burden",
        "sleep_quality_raw": "poor_sleep_burden",
        "soreness_raw": "soreness_burden",
        "stress_raw": "stress_burden",
    }
    for raw_field, burden_field in burden_mapping.items():
        merged[burden_field] = (5.0 - merged[raw_field]) / 4.0

    return merged


def load_game(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "player_name",
        "team_performance",
        "offensive_performance",
        "defensive_performance",
        "timestamp",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"game-performance.csv missing {sorted(required - set(frame.columns))}")
    frame["date"] = pd.to_datetime(
        frame["timestamp"], format="%d.%m.%Y", errors="coerce"
    ).dt.normalize()
    if frame["date"].isna().any():
        raise ValueError("Invalid DD.MM.YYYY timestamp in game-performance.csv")
    original = frame.drop(columns=["date"]).columns.tolist()
    frame = frame.drop_duplicates(subset=original, keep="first")
    if frame.duplicated(["player_name", "date"]).any():
        raise AssertionError("Non-identical duplicate game player-date records")
    for field in [
        "team_performance",
        "offensive_performance",
        "defensive_performance",
    ]:
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    frame["has_game_record"] = np.int8(1)
    return frame.drop(columns=["timestamp"])


def load_illness(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"player_name", "timestamp"}
    if not required.issubset(frame.columns):
        raise ValueError(f"illness.csv missing {sorted(required - set(frame.columns))}")
    frame["date"] = pd.to_datetime(
        frame["timestamp"], format="%d.%m.%Y", errors="coerce"
    ).dt.normalize()
    if frame["date"].isna().any():
        raise ValueError("Invalid DD.MM.YYYY timestamp in illness.csv")
    daily = (
        frame.groupby(["player_name", "date"], as_index=False)
        .size()
        .rename(columns={"size": "illness_record_count"})
    )
    daily["has_illness_record"] = np.int8(1)
    return daily


def load_gps(path: Path) -> pd.DataFrame:
    """Read only the verified daily GPS cache; never scan raw Parquet here."""
    if path.name != "source_gps_daily.csv":
        raise AssertionError("Only verified source_gps_daily.csv is permitted")
    gps = pd.read_csv(path)
    required = {"player_name", "date"}
    if not required.issubset(gps.columns):
        raise ValueError(f"GPS cache missing {sorted(required - set(gps.columns))}")
    gps["date"] = pd.to_datetime(
        gps["date"], format="%Y-%m-%d", errors="raise"
    ).dt.normalize()
    gps["player_name"] = gps["player_name"].astype(str).str.strip()
    if gps.duplicated(["player_name", "date"]).any():
        raise AssertionError("Duplicate keys in verified GPS daily cache")
    gps["has_gps_record"] = np.int8(1)
    return gps


def load_locked_injury(path: Path) -> pd.DataFrame:
    """Load only the locked report-day outcome, not location/severity fields."""
    injury = pd.read_csv(path)
    expected = ["player_name", "date", "injury_report"]
    if list(injury.columns) != expected:
        raise AssertionError(f"Locked injury schema changed: {injury.columns.tolist()}")
    injury["date"] = pd.to_datetime(
        injury["date"], format="%Y-%m-%d", errors="raise"
    ).dt.normalize()
    injury["injury_report"] = pd.to_numeric(
        injury["injury_report"], errors="raise"
    ).astype("int8")
    if injury.duplicated(["player_name", "date"]).any():
        raise AssertionError("Duplicate keys in locked injury report-day file")
    return injury


def mapping_row(
    field: str,
    role: str,
    source: str,
    source_field: str,
    transformation: str,
    origin: bool,
    do_not_use: bool = False,
) -> dict[str, Any]:
    return {
        "panel_field": field,
        "role": role,
        "source_file": source,
        "source_field": source_field,
        "transformation": transformation,
        "defines_prediction_origin": int(origin),
        "do_not_use_as_predictor": int(do_not_use),
    }


def build_input_paths() -> list[Path]:
    paths = [
        SUBJECTIVE / "training-load/session.json",
        SUBJECTIVE / "game-performance/game-performance.csv",
        SUBJECTIVE / "illness/illness.csv",
        GPS_DAILY,
        INJURY_REPORT_DAY,
    ]
    paths.extend(
        SUBJECTIVE / "training-load" / f"{name}.csv"
        for name in TRAINING_EXPORTS
    )
    paths.extend(
        SUBJECTIVE / "wellness" / f"{name}.csv"
        for name in WELLNESS_RAW
    )
    return paths


def main() -> None:
    print("=" * 72)
    print("01: SoccerMon主客观数据预处理与运动员—日主表构建")
    print("=" * 72)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    input_paths = build_input_paths()
    missing = [str(path) for path in input_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少必要输入文件:\n" + "\n".join(missing))

    print("[1/7] 读取并聚合训练负荷...")
    sessions = load_sessions(SUBJECTIVE / "training-load/session.json")
    training_exports = {
        field_name: parse_wide_ddmmyyyy(
            SUBJECTIVE / "training-load" / f"{source_name}.csv", field_name
        )
        for source_name, field_name in TRAINING_EXPORTS.items()
    }
    print(f"  session日级记录: {len(sessions):,}")
    print(f"  源域训练负荷CSV: {len(training_exports)}个")

    print("[2/7] 读取wellness、比赛表现和疾病报告...")
    wellness = load_wellness()
    # The exported wellness files are rectangular date-by-player tables.
    # A row with no reported wellness value is not a real PmSys interaction
    # and must not create a prediction origin.
    wellness = wellness.loc[wellness["has_wellness_record"].eq(1)].copy()
    game = load_game(SUBJECTIVE / "game-performance/game-performance.csv")
    illness = load_illness(SUBJECTIVE / "illness/illness.csv")
    print(f"  wellness日级记录: {len(wellness):,}")
    print(f"  game日级记录: {len(game):,}")
    print(f"  illness日级记录: {len(illness):,}")

    print("[3/7] 读取已验证GPS日级缓存和锁定结局...")
    gps = load_gps(GPS_DAILY)
    injury = load_locked_injury(INJURY_REPORT_DAY)
    print(f"  GPS日级记录: {len(gps):,}")
    print(f"  锁定injury report-days: {len(injury):,}")

    # PmSys cohort eligibility is based only on actual PmSys report sources.
    # Derived training-load exports are predictors, not cohort evidence.
    print("[4/7] 建立PmSys运动员—年度资格队列和prediction-origin主键...")
    subjective_sources = [
        sessions[["player_name", "date"]],
        wellness[["player_name", "date"]],
        game[["player_name", "date"]],
        illness[["player_name", "date"]],
        injury[["player_name", "date"]],
    ]
    subjective_dates = pd.concat(subjective_sources, ignore_index=True).drop_duplicates()
    subjective_dates["calendar_year"] = subjective_dates["date"].dt.year.astype(int)
    subjective_player_year = subjective_dates[["player_name", "calendar_year"]].drop_duplicates()

    gps_dates = gps[["player_name", "date"]].copy()
    gps_dates["calendar_year"] = gps_dates["date"].dt.year.astype(int)
    all_player_year = pd.concat(
        [subjective_player_year, gps_dates[["player_name", "calendar_year"]]],
        ignore_index=True,
    ).drop_duplicates()
    cohort = all_player_year.merge(
        subjective_player_year.assign(pmsys_eligible=np.int8(1)),
        on=["player_name", "calendar_year"],
        how="left",
        validate="one_to_one",
    )
    cohort["pmsys_eligible"] = cohort["pmsys_eligible"].fillna(0).astype("int8")

    # The raw union retains every observed source key, including injury-only
    # dates.  Derived rectangular training exports do not create keys.
    raw_union_keys = pd.concat(
        [*subjective_sources, gps[["player_name", "date"]]],
        ignore_index=True,
    ).drop_duplicates()
    raw_union_keys["calendar_year"] = raw_union_keys["date"].dt.year.astype(int)
    raw_union_keys = raw_union_keys.merge(
        cohort,
        on=["player_name", "calendar_year"],
        how="left",
        validate="many_to_one",
    )

    # A prediction origin requires a real session, wellness, game, or GPS row.
    # Illness/injury-only dates remain in raw_union_keys but do not independently
    # create a primary-model origin because they contain no monitoring inputs.
    origin_keys = pd.concat(
        [
            sessions[["player_name", "date"]],
            wellness[["player_name", "date"]],
            game[["player_name", "date"]],
            gps[["player_name", "date"]],
        ],
        ignore_index=True,
    ).drop_duplicates()
    origin_keys["calendar_year"] = origin_keys["date"].dt.year.astype(int)
    origin_keys = origin_keys.merge(
        cohort,
        on=["player_name", "calendar_year"],
        how="left",
        validate="many_to_one",
    )
    origin_keys = origin_keys.loc[origin_keys["pmsys_eligible"].eq(1)].copy()
    origin_keys = origin_keys.sort_values(["player_name", "date"]).reset_index(drop=True)
    if origin_keys.duplicated(["player_name", "date"]).any():
        raise AssertionError("Duplicate prediction-origin keys")
    print(f"  PmSys合格运动员—年度组合: {len(cohort):,}")
    print(f"  raw source-key union: {len(raw_union_keys):,}")
    print(f"  prediction origins: {len(origin_keys):,}")

    print("[5/7] 按player_name + date左连接所有主客观日级数据...")
    raw_panel = raw_union_keys[["player_name", "date", "calendar_year", "pmsys_eligible"]].copy()
    raw_panel = raw_panel.merge(
        sessions.drop(columns=["has_training_record"]),
        on=["player_name", "date"],
        how="left",
        validate="one_to_one",
    )
    raw_panel["has_training_record"] = raw_panel[[
        "session_count", "session_rpe_mean", "session_srpe_sum_au",
        "session_duration_sum_min",
    ]].notna().any(axis=1).astype("int8")
    raw_panel = raw_panel.merge(wellness, on=["player_name", "date"], how="left", validate="one_to_one")
    raw_panel = raw_panel.merge(game, on=["player_name", "date"], how="left", validate="one_to_one")
    raw_panel = raw_panel.merge(illness, on=["player_name", "date"], how="left", validate="one_to_one")

    for availability_field in [
        "has_training_record",
        "has_wellness_record",
        "has_game_record",
        "has_illness_record",
        "has_gps_record",
    ]:
        if availability_field in raw_panel.columns:
            raw_panel[availability_field] = raw_panel[availability_field].fillna(0).astype("int8")

    for exported in training_exports.values():
        raw_panel = raw_panel.merge(
            exported, on=["player_name", "date"], how="left", validate="one_to_one"
        )
    raw_panel = raw_panel.merge(gps, on=["player_name", "date"], how="left", validate="one_to_one")
    raw_panel["has_gps_record"] = raw_panel["has_gps_record"].fillna(0).astype("int8")

    raw_panel["origin_eligible"] = (
        raw_panel[[
            "has_training_record",
            "has_wellness_record",
            "has_game_record",
            "has_gps_record",
        ]].any(axis=1).astype("int8")
    )

    # Carry only the locked daily report outcome as label/QC metadata.
    raw_panel = raw_panel.merge(
        injury.rename(columns={"injury_report": "injury_report_t"}),
        on=["player_name", "date"],
        how="left",
        validate="one_to_one",
    )
    raw_panel["injury_report_t"] = raw_panel["injury_report_t"].fillna(0).astype("int8")
    raw_panel["target_date"] = raw_panel["date"] + pd.Timedelta(days=1)
    target_lookup = injury.rename(
        columns={"date": "target_date", "injury_report": "target_injury_report"}
    )
    raw_panel = raw_panel.merge(
        target_lookup[["player_name", "target_date", "target_injury_report"]],
        on=["player_name", "target_date"],
        how="left",
        validate="one_to_one",
    )

    monitoring_dates = pd.concat(
        [origin_keys[["player_name", "date"]], injury[["player_name", "date"]]],
        ignore_index=True,
    )
    monitoring_last = (
        monitoring_dates.groupby("player_name", as_index=False)["date"].max()
        .rename(columns={"date": "monitoring_last_date"})
    )
    raw_panel = raw_panel.merge(monitoring_last, on="player_name", how="left", validate="many_to_one")
    raw_panel["outcome_boundary_flag"] = (
        raw_panel["target_date"] > raw_panel["monitoring_last_date"]
    ).astype("int8")
    raw_panel["y_t1"] = raw_panel["target_injury_report"].fillna(0).astype("Int8")
    raw_panel.loc[raw_panel["outcome_boundary_flag"].eq(1), "y_t1"] = pd.NA
    raw_panel = raw_panel.drop(columns=["target_injury_report"])
    raw_panel["origin_eligibility_reason"] = np.select(
        [
            raw_panel["origin_eligible"].eq(1),
            raw_panel["injury_report_t"].eq(1),
            raw_panel["has_illness_record"].eq(1),
        ],
        ["monitoring_observed", "injury_only", "illness_only"],
        default="other_source_only",
    )
    raw_panel["team"] = raw_panel["player_name"].str.extract(r"^(Team[^-]+)", expand=False)
    raw_panel = raw_panel.sort_values(["player_name", "date"]).reset_index(drop=True)

    # This is the modeling risk set.  The complete raw union above is retained
    # separately so injury-only dates are auditable rather than silently deleted.
    panel = raw_panel.loc[
        raw_panel["pmsys_eligible"].eq(1) & raw_panel["origin_eligible"].eq(1)
    ].copy()
    panel = panel.sort_values(["player_name", "date"]).reset_index(drop=True)

    if raw_panel.duplicated(["player_name", "date"]).any():
        raise AssertionError("Raw merged panel has duplicate player_name + date keys")
    if not raw_panel.columns.is_unique:
        raise AssertionError("Raw merged panel has duplicate column names")
    if not raw_panel["target_date"].eq(raw_panel["date"] + pd.Timedelta(days=1)).all():
        raise AssertionError("target_date is not exact calendar date + 1")

    # No GPS imputation and no global imputation of any predictor field.
    gps_fields = [
        field for field in gps.columns
        if field not in {"player_name", "date", "has_gps_record"}
    ]
    no_gps = raw_panel["has_gps_record"].fillna(0).eq(0)
    if raw_panel.loc[no_gps, gps_fields].notna().any().any():
        raise AssertionError("GPS values appeared where the GPS cache had no row")

    print("[6/7] 生成字段映射、缺失审计和质量报告...")
    outcome_fields = {
        "injury_report_t",
        "target_date",
        "y_t1",
        "outcome_boundary_flag",
        "monitoring_last_date",
        "pmsys_eligible",
        "calendar_year",
    }
    identifier_fields = {"player_name", "date", "team"}
    qc_fields = {
        "has_training_record",
        "has_wellness_record",
        "has_game_record",
        "has_illness_record",
        "has_gps_record",
        "origin_eligible",
    }
    mapping: list[dict[str, Any]] = []
    mapping.extend([
        mapping_row("player_name", "identifier", "all sources", "player_name", "identity", False, True),
        mapping_row("date", "identifier", "all sources", "date/timestamp", "source-specific parsing", False, True),
        mapping_row("calendar_year", "cohort_qc", "derived", "date", "calendar year", False, True),
        mapping_row("pmsys_eligible", "cohort_qc", "real PmSys reports", "player + year", "subjective report presence", False, True),
        mapping_row("team", "metadata", "player_name", "player_name prefix", "Team prefix", False, True),
    ])
    for field in [
        "session_count", "session_rpe_mean", "session_srpe_sum_au", "session_duration_sum_min"
    ]:
        mapping.append(mapping_row(field, "predictor", "data/subjective/training-load/session.json", field, "session-level records aggregated by player-day; sums use min_count=1", True))
    for source_name, field_name in TRAINING_EXPORTS.items():
        mapping.append(mapping_row(field_name, "source_native_predictor", f"data/subjective/training-load/{source_name}.csv", source_name, "wide-to-long; source value preserved; does not create origin", False))
    for source_name, field_name in WELLNESS_RAW.items():
        mapping.append(mapping_row(field_name, "predictor", f"data/subjective/wellness/{source_name}.csv", source_name, "wide-to-long; raw source scale preserved", True))
    for field in ["fatigue_burden", "poor_sleep_burden", "soreness_burden", "stress_burden"]:
        mapping.append(mapping_row(field, "harmonized_predictor", "data/subjective/wellness/*.csv", field, "(5 - raw) / 4; no imputation", True))
    for field in ["team_performance", "offensive_performance", "defensive_performance"]:
        mapping.append(mapping_row(field, "predictor", "data/subjective/game-performance/game-performance.csv", field, "one daily game-performance record", True))
    for field in ["illness_record_count"]:
        mapping.append(mapping_row(field, "predictor", "data/subjective/illness/illness.csv", field, "daily record count", False))
    for field in gps_fields:
        role = "gps_qc" if field.startswith("gps_") or field.endswith("_count") else "gps_predictor"
        mapping.append(mapping_row(field, role, str(GPS_DAILY.relative_to(ROOT)), field, "verified daily GPS cache; missing retained as NA", True, role == "gps_qc"))
    for field in sorted(qc_fields):
        if field in panel.columns:
            mapping.append(mapping_row(field, "availability_qc", "derived source presence", field, "binary source-row presence; not imputed", False, True))
    mapping.append(mapping_row(
        "origin_eligibility_reason",
        "origin_qc",
        "derived source presence",
        "source flags + injury_report_t",
        "monitoring_observed, injury_only, illness_only, or other_source_only",
        False,
        True,
    ))
    mapping.extend([
        mapping_row("injury_report_t", "outcome_qc", str(INJURY_REPORT_DAY.relative_to(ROOT)), "injury_report", "same-day exact player-date match", False, True),
        mapping_row("target_date", "outcome_qc", "derived", "date", "date + 1 calendar day", False, True),
        mapping_row("y_t1", "outcome", str(INJURY_REPORT_DAY.relative_to(ROOT)), "injury_report", "exact player + target_date match", False, True),
        mapping_row("monitoring_last_date", "outcome_qc", "panel + locked report days", "date", "per-player maximum observed date", False, True),
        mapping_row("outcome_boundary_flag", "outcome_qc", "derived", "target_date and monitoring_last_date", "target beyond last observed date", False, True),
    ])
    mapping_frame = pd.DataFrame(mapping).drop_duplicates("panel_field", keep="last")
    missing_mapping = sorted(set(panel.columns) - set(mapping_frame["panel_field"]))
    if missing_mapping:
        raise AssertionError(f"Fields missing from SOURCE_FIELD_MAPPING: {missing_mapping}")

    predictor_fields = [
        field for field in panel.columns
        if field not in DO_NOT_USE_AS_PREDICTOR
        and field not in qc_fields
        and field not in identifier_fields
        and field not in outcome_fields
    ]
    missingness = pd.DataFrame({
        "field": predictor_fields,
        "role": [mapping_frame.set_index("panel_field").loc[field, "role"] for field in predictor_fields],
        "rows": len(panel),
        "missing_n": [int(panel[field].isna().sum()) for field in predictor_fields],
        "missing_rate": [float(panel[field].isna().mean()) for field in predictor_fields],
    }).sort_values(["missing_rate", "field"], ascending=[False, True])

    gps_with_year = gps[["player_name", "date"]].copy()
    gps_with_year["calendar_year"] = gps_with_year["date"].dt.year.astype(int)
    gps_with_year = gps_with_year.merge(
        cohort[["player_name", "calendar_year", "pmsys_eligible"]],
        on=["player_name", "calendar_year"],
        how="left",
        validate="many_to_one",
    )
    cohort_audit = pd.DataFrame([
        {"metric": "subjective_PmSys_players", "value": subjective_dates["player_name"].nunique()},
        {"metric": "GPS_players", "value": gps["player_name"].nunique()},
        {"metric": "union_players", "value": len(set(subjective_dates["player_name"]) | set(gps["player_name"]))},
        {"metric": "intersection_players", "value": len(set(subjective_dates["player_name"]) & set(gps["player_name"]))},
        {"metric": "GPS_ineligible_athlete_days_excluded_by_cohort", "value": int(gps_with_year["pmsys_eligible"].fillna(0).eq(0).sum())},
    ])

    origin_eligibility_audit = (
        raw_panel.groupby(
            ["pmsys_eligible", "origin_eligible", "origin_eligibility_reason"],
            dropna=False,
        )
        .size()
        .reset_index(name="rows")
        .sort_values(
            ["pmsys_eligible", "origin_eligible", "origin_eligibility_reason"],
            ascending=[False, False, True],
        )
    )

    outcome_summary = pd.DataFrame([
        {"metric": "raw_source_union_rows", "value": len(raw_panel)},
        {"metric": "raw_source_union_players", "value": raw_panel["player_name"].nunique()},
        {"metric": "raw_injury_only_rows", "value": int(raw_panel["origin_eligibility_reason"].eq("injury_only").sum())},
        {"metric": "raw_illness_only_rows", "value": int(raw_panel["origin_eligibility_reason"].eq("illness_only").sum())},
        {"metric": "raw_non_origin_rows", "value": int(raw_panel["origin_eligible"].eq(0).sum())},
        {"metric": "prediction_origins", "value": len(panel)},
        {"metric": "players", "value": panel["player_name"].nunique()},
        {"metric": "y_t1_positive_non_boundary", "value": int(panel.loc[panel["outcome_boundary_flag"].eq(0), "y_t1"].sum())},
        {"metric": "outcome_evaluable_rows", "value": int(panel["y_t1"].notna().sum())},
        {"metric": "y_t1_prevalence_non_boundary", "value": float(panel.loc[panel["outcome_boundary_flag"].eq(0), "y_t1"].mean())},
        {"metric": "outcome_boundary_rows", "value": int(panel["outcome_boundary_flag"].sum())},
        {"metric": "injury_report_days", "value": len(injury)},
        {"metric": "gps_record_coverage", "value": float(panel["has_gps_record"].mean())},
        {"metric": "duplicate_player_date_keys", "value": int(panel.duplicated(["player_name", "date"]).sum())},
    ])

    print("[7/7] 保存正式预处理输出...")
    raw_csv_path = OUTPUT_DIR / "source_athlete_day_raw_union.csv"
    csv_path = OUTPUT_DIR / "source_athlete_day_preprocessed.csv"
    raw_parquet_path = OUTPUT_DIR / "source_athlete_day_raw_union.parquet"
    parquet_path = OUTPUT_DIR / "source_athlete_day_preprocessed.parquet"
    raw_panel.to_csv(raw_csv_path, index=False, date_format="%Y-%m-%d")
    panel.to_csv(csv_path, index=False, date_format="%Y-%m-%d")
    raw_parquet_written = False
    parquet_written = False
    try:
        raw_panel.to_parquet(raw_parquet_path, index=False)
        panel.to_parquet(parquet_path, index=False)
        raw_parquet_written = True
        parquet_written = True
    except (ImportError, ModuleNotFoundError) as exc:
        print(f"  警告：当前环境缺少Parquet写入引擎，已保留CSV输出: {exc}")

    mapping_frame.sort_values(["role", "panel_field"]).to_csv(
        OUTPUT_DIR / "SOURCE_FIELD_MAPPING.csv", index=False
    )
    missingness.to_csv(OUTPUT_DIR / "SOURCE_MISSINGNESS.csv", index=False)
    cohort.to_csv(OUTPUT_DIR / "PMSYS_PLAYER_YEAR.csv", index=False)
    origin_eligibility_audit.to_csv(
        OUTPUT_DIR / "SOURCE_ORIGIN_ELIGIBILITY_AUDIT.csv", index=False
    )
    cohort_audit.to_csv(OUTPUT_DIR / "PMSYS_COHORT_AUDIT.csv", index=False)
    outcome_summary.to_csv(OUTPUT_DIR / "SOURCE_OUTCOME_SUMMARY.csv", index=False)

    report = [
        "# SoccerMon source athlete-day preprocessing report",
        "",
        "本程序完成原始来源键联合、日级数据清洗、时间格式统一和主客观数据合并。",
        "未进行缺失值插补、标准化、窗口特征、特征选择、EDA或模型训练。",
        "",
        f"- raw source-key union rows: {len(raw_panel):,}",
        f"- raw source-key union players: {raw_panel['player_name'].nunique()}",
        f"- raw injury-only rows retained: {int(raw_panel['origin_eligibility_reason'].eq('injury_only').sum()):,}",
        f"- raw illness-only rows retained: {int(raw_panel['origin_eligibility_reason'].eq('illness_only').sum()):,}",
        f"- output rows: {len(panel):,}",
        f"- players: {panel['player_name'].nunique()}",
        f"- date range: {panel['date'].min().date()} to {panel['date'].max().date()}",
        f"- y_t1 positives among non-boundary rows: {int(panel.loc[panel['outcome_boundary_flag'].eq(0), 'y_t1'].sum())}",
        f"- outcome-evaluable rows: {int(panel['y_t1'].notna().sum()):,}",
        f"- outcome boundary rows: {int(panel['outcome_boundary_flag'].sum()):,}",
        f"- GPS record coverage: {panel['has_gps_record'].mean():.2%}",
        f"- duplicate player_name + date keys: {int(panel.duplicated(['player_name', 'date']).sum())}",
        "",
        "## Input sources",
        "",
        "- training-load/session.json: session-level records aggregated to player-day.",
        "- training-load/*.csv: daily/rolling source-native exports retained as fields; they do not create prediction origins.",
        "- wellness/*.csv: all seven source wellness files reshaped from wide to long.",
        "- game-performance/game-performance.csv and illness/illness.csv: date-normalized daily reports.",
        "- verified_outputs/04_source_gps/source_gps_daily.csv: verified daily GPS cache; raw Parquet was not read.",
        "- verified_outputs/07_source_injury_reportday_t1_v1/04_injury_report_day.csv: locked outcome/QC only.",
        "",
        "## Integrity decisions",
        "",
        "- Join key: player_name + date.",
        "- Raw union: session, wellness, game, illness, GPS, and locked injury report-day keys are retained for audit.",
        "- Cohort eligibility: actual PmSys report presence by player_name + calendar_year.",
        "- Prediction origins: session, wellness, game-performance, or verified GPS presence.",
        "- Injury-only and illness-only dates are retained in the raw union but are not primary prediction origins.",
        "- GPS missing values remain NA; no unconditional zero filling was performed.",
        "- Session daily sums use min_count=1; an existing session day with all numeric values missing remains NA.",
        "- injury_report_t and y_t1 are outcome/QC fields and are forbidden predictors.",
        "- No rolling windows were created in this stage.",
        "- No imputation, standardization, feature selection, EDA, or model fitting was performed.",
        "",
        "## Output",
        "",
        "- source_athlete_day_raw_union.csv: written",
        f"- source_athlete_day_preprocessed.csv: written",
        f"- source_athlete_day_raw_union.parquet: {'written' if raw_parquet_written else 'not written; install a Parquet engine before formal use'}",
        f"- source_athlete_day_preprocessed.parquet: {'written' if parquet_written else 'not written; install a Parquet engine before formal use'}",
    ]
    (OUTPUT_DIR / "SOURCE_PREPROCESSING_REPORT.md").write_text(
        "\n".join(report), encoding="utf-8"
    )

    output_files = [
        "source_athlete_day_raw_union.csv",
        "source_athlete_day_preprocessed.csv",
        "SOURCE_FIELD_MAPPING.csv",
        "SOURCE_MISSINGNESS.csv",
        "PMSYS_PLAYER_YEAR.csv",
        "SOURCE_ORIGIN_ELIGIBILITY_AUDIT.csv",
        "PMSYS_COHORT_AUDIT.csv",
        "SOURCE_OUTCOME_SUMMARY.csv",
        "SOURCE_PREPROCESSING_REPORT.md",
    ]
    if raw_parquet_written:
        output_files.append("source_athlete_day_raw_union.parquet")
    if parquet_written:
        output_files.append("source_athlete_day_preprocessed.parquet")
    manifest = {
        "script": Path(__file__).name,
        "generated_at": datetime.now().astimezone(),
        "analysis_unit": "raw source-key union plus PmSys-eligible prediction-origin player-day",
        "join_key": ["player_name", "date"],
        "input_sources": [str(path.relative_to(ROOT)) for path in input_paths],
        "raw_gps_parquet_read": False,
        "gps_source": str(GPS_DAILY.relative_to(ROOT)),
        "locked_outcome_source": str(INJURY_REPORT_DAY.relative_to(ROOT)),
        "no_imputation": True,
        "no_standardization": True,
        "no_window_features": True,
        "no_feature_selection": True,
        "no_eda": True,
        "no_model_training": True,
        "raw_union_rows": len(raw_panel),
        "raw_injury_only_rows_retained": int(raw_panel["origin_eligibility_reason"].eq("injury_only").sum()),
        "raw_illness_only_rows_retained": int(raw_panel["origin_eligibility_reason"].eq("illness_only").sum()),
        "prediction_origin_rows": len(panel),
        "do_not_use_as_predictor": sorted(DO_NOT_USE_AS_PREDICTOR),
        "raw_parquet_written": raw_parquet_written,
        "parquet_written": parquet_written,
        "input_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in input_paths},
        "output_sha256": {name: sha256_file(OUTPUT_DIR / name) for name in output_files},
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(native(manifest), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n01预处理完成")
    print(f"  原始联合主表: {raw_csv_path}")
    print(f"  建模起点表: {csv_path}")
    print(f"  原始联合行数: {len(raw_panel):,}（保留伤病-only日期）")
    print(f"  行数: {len(panel):,} | 运动员: {panel['player_name'].nunique()}")
    print(f"  GPS覆盖率: {panel['has_gps_record'].mean():.2%}")
    print(f"  输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
