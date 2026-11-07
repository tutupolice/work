#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地目标域运动员—自然日数据重建与标签审计。

输入：当前附件中的基线问卷、晨间问卷、训练后问卷、GNSS/IMU原始文件和队医伤病表。
输出：原始文件审计、运动员—日主表、伤病标签审计及可追溯的处理说明。

本程序不构造未来窗口标签，不把缺失报告改写成0，不把同日伤病行误当成多个运动员—日阳性。
GNSS/IMU日期以文件内部时间为主，文件夹日期只作为一致性审计字段。
"""

from __future__ import annotations

import hashlib
import os
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(os.environ.get("LOCAL_DATA_ROOT", Path(__file__).resolve().parents[1]))
RAW_OBJECTIVE = BASE / "00_原始客观数据"
SUBJECTIVE = BASE / "主观数据"
INJURY_FILE = BASE / "女足损伤记录表.xlsx"
BASELINE_FILE = BASE / "东北师范大学女足基线健康与既往损伤史调查问卷xlsx.xlsx"
AUDIT = BASE / "13_目标域原始审计"
OUT = BASE / "14_目标域日级重建"

BRUMS_MAP = {
    "完全没有": 0,
    "有一点": 1,
    "中等程度": 2,
    "比较强烈": 3,
    "非常强烈": 4,
}


def text(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", "", str(value)).replace("　", "")


def name_key(value) -> str:
    s = text(value)
    s = re.sub(r"\s*\d{1,2}\.\d{2}\.\d{2}$", "", s)
    s = re.sub(r"\(\d+\)$", "", s)
    s = re.sub(r"^\d{1,2}号[-—_]", "", s)
    s = re.sub(r"^\d{1,2}[-—_]", "", s)
    return s


def date_value(value) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT
    parsed = pd.to_datetime(str(value).strip(), errors="coerce")
    return parsed.normalize() if not pd.isna(parsed) else pd.NaT


def file_date(path: Path) -> pd.Timestamp:
    month = re.search(r"(\d{1,2})月", path.parent.name)
    day = re.match(r"^(\d{1,2})\.(\d{1,2})", path.stem)
    if not month or not day:
        return pd.NaT
    return pd.Timestamp(2026, int(month.group(1)), int(day.group(2)))


def folder_date(path: Path) -> pd.Timestamp:
    month = re.search(r"(\d{1,2})月", path.parent.name)
    day = re.match(r"^(\d{1,2})\.(\d{1,2})", path.name)
    if not month or not day:
        return pd.NaT
    return pd.Timestamp(2026, int(month.group(1)), int(day.group(2)))


def numeric(value):
    if pd.isna(value):
        return np.nan
    m = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
    return float(m.group(0)) if m else np.nan


def first_col(df: pd.DataFrame, tokens: list[str]) -> str | None:
    for col in df.columns:
        if all(token in str(col) for token in tokens):
            return col
    return None


def read_baseline() -> tuple[pd.DataFrame, set[str]]:
    df = pd.read_excel(BASELINE_FILE)
    col = first_col(df, ["姓名", "设备编号"])
    if col is None:
        raise ValueError("基线问卷缺少姓名与设备编号字段")
    out = pd.DataFrame({"player_id_raw": df[col].astype(str)})
    out["player_name"] = out["player_id_raw"].map(name_key)
    out = out.drop_duplicates("player_name").sort_values("player_name")
    return out, set(out["player_name"])


def audit_objective(players: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    file_rows = []
    daily_rows = []
    for path in sorted(RAW_OBJECTIVE.rglob("*.xlsx")):
        raw_stem = path.stem
        player = name_key(raw_stem)
        fdate = folder_date(path.parent)
        rec = {
            "source_file": str(path),
            "raw_file_name": path.name,
            "player_name": player,
            "folder_date": fdate,
            "folder_date_str": fdate.strftime("%Y-%m-%d") if not pd.isna(fdate) else "",
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        try:
            raw = pd.read_excel(path)
            rec["raw_rows"] = int(len(raw))
            time_col = "时间" if "时间" in raw.columns else None
            if time_col is None:
                rec.update({"valid_time_rows": 0, "content_date": pd.NaT, "content_date_count": 0})
                file_rows.append(rec)
                continue
            times = pd.to_datetime(raw[time_col], errors="coerce")
            valid = times.dropna()
            content_hash = hashlib.sha256(pd.util.hash_pandas_object(raw, index=True).values.tobytes()).hexdigest()
            date_counts = valid.dt.normalize().value_counts()
            cdate = date_counts.index[0] if len(date_counts) else pd.NaT
            rec.update(
                {
                    "valid_time_rows": int(valid.size),
                    "content_hash": content_hash,
                    "content_date": cdate,
                    "content_date_count": int(date_counts.iloc[0]) if len(date_counts) else 0,
                    "content_date_nunique": int(valid.dt.normalize().nunique()),
                    "time_min": valid.min() if len(valid) else pd.NaT,
                    "time_max": valid.max() if len(valid) else pd.NaT,
                    "folder_content_date_match": bool(pd.notna(cdate) and cdate == fdate),
                }
            )
            speed_col = next((c for c in raw.columns if "速度" in str(c)), None)
            hr_col = "心率" if "心率" in raw.columns else None
            sat_col = "卫星数量" if "卫星数量" in raw.columns else None
            signal_col = "信号质量" if "信号质量" in raw.columns else None
            hdop_col = "HDOP" if "HDOP" in raw.columns else None
            speed = pd.to_numeric(raw[speed_col], errors="coerce") if speed_col else pd.Series(dtype=float)
            hr = pd.to_numeric(raw[hr_col], errors="coerce") if hr_col else pd.Series(dtype=float)
            sat = pd.to_numeric(raw[sat_col], errors="coerce") if sat_col else pd.Series(dtype=float)
            signal = pd.to_numeric(raw[signal_col], errors="coerce") if signal_col else pd.Series(dtype=float)
            hdop = pd.to_numeric(raw[hdop_col], errors="coerce") if hdop_col else pd.Series(dtype=float)
            cdate_str = cdate.strftime("%Y-%m-%d") if pd.notna(cdate) else ""
            daily_rows.append(
                {
                    "player_name": player,
                    "date": cdate,
                    "source_file": str(path),
                    "raw_rows": int(len(raw)),
                    "valid_time_rows": int(valid.size),
                    "time_min": valid.min() if len(valid) else pd.NaT,
                    "time_max": valid.max() if len(valid) else pd.NaT,
                    "session_duration_min": (valid.max() - valid.min()).total_seconds() / 60 if len(valid) else np.nan,
                    "speed_mean_ms": float(speed.mean()) if speed.notna().any() else np.nan,
                    "speed_max_ms": float(speed.max()) if speed.notna().any() else np.nan,
                    "speed_valid_rate": float(speed.notna().mean()) if len(raw) else np.nan,
                    "hr_mean_bpm": float(hr.mean()) if hr.notna().any() else np.nan,
                    "hr_max_bpm": float(hr.max()) if hr.notna().any() else np.nan,
                    "satellite_mean": float(sat.mean()) if sat.notna().any() else np.nan,
                    "signal_quality_mean": float(signal.mean()) if signal.notna().any() else np.nan,
                    "hdop_mean": float(hdop.mean()) if hdop.notna().any() else np.nan,
                    "folder_date": fdate,
                    "folder_content_date_match": bool(pd.notna(cdate) and cdate == fdate),
                    "file_sha256": rec["file_sha256"],
                    "content_hash": content_hash,
                    "content_date_str": cdate_str,
                }
            )
        except Exception as exc:
            rec["read_error"] = str(exc)
        file_rows.append(rec)

    files = pd.DataFrame(file_rows)
    sessions = pd.DataFrame(daily_rows)
    if sessions.empty:
        return files, sessions
    sessions["athlete_day_key"] = sessions["player_name"] + "|" + sessions["content_date_str"]
    duplicate_keys = sessions["athlete_day_key"].value_counts()
    sessions["objective_file_count"] = sessions["athlete_day_key"].map(duplicate_keys)
    sessions["objective_duplicate_day"] = (sessions["objective_file_count"] > 1).astype(int)
    sessions["exact_binary_duplicate"] = sessions.groupby("athlete_day_key")["content_hash"].transform("nunique").eq(1).astype(int)
    sessions["selected_for_daily_aggregation"] = 1
    exact_groups = sessions[sessions["exact_binary_duplicate"] == 1].groupby("athlete_day_key").groups
    for _, indices in exact_groups.items():
        ordered = sorted(indices, key=lambda i: sessions.loc[i, "source_file"])
        sessions.loc[ordered[1:], "selected_for_daily_aggregation"] = 0
    sessions["objective_day_resolution"] = np.select(
        [sessions["objective_file_count"].eq(1), sessions["exact_binary_duplicate"].eq(1)],
        ["single_file", "exact_duplicate_keep_one"],
        default="multiple_files_manual_review",
    )
    files.to_csv(OUT / "objective_file_audit.csv", index=False, encoding="utf-8-sig")
    sessions.to_csv(OUT / "objective_session_audit.csv", index=False, encoding="utf-8-sig")
    sessions[["player_name", "date", "source_file", "objective_file_count", "exact_binary_duplicate", "selected_for_daily_aggregation", "objective_day_resolution"]].to_csv(
        OUT / "objective_session_resolution.csv", index=False, encoding="utf-8-sig"
    )
    return files, sessions


def audit_morning(players: set[str]) -> pd.DataFrame:
    rows = []
    for path in sorted((SUBJECTIVE / "晨间").rglob("*.xlsx")):
        raw = pd.read_excel(path)
        name_col = first_col(raw, ["姓名与球衣号"]) or first_col(raw, ["姓名"])
        date_col = first_col(raw, ["填写日期"])
        if not name_col or not date_col:
            continue
        for _, r in raw.iterrows():
            row = {
                "player_name": name_key(r.get(name_col)),
                "date": date_value(r.get(date_col)),
                "source_file": str(path),
                "sleep_quality_1_7": numeric(r.get(first_col(raw, ["睡眠质量"]))),
                "morning_fatigue_1_7": numeric(r.get(first_col(raw, ["疲劳感"]))),
                "muscle_soreness_1_7": numeric(r.get(first_col(raw, ["肌肉酸痛"]))),
                "mental_stress_1_7": numeric(r.get(first_col(raw, ["心理压力"]))),
                "menstrual_flag": text(r.get(first_col(raw, ["是否处于生理期"]))) if first_col(raw, ["是否处于生理期"]) else "",
                "menstrual_discomfort": numeric(r.get(first_col(raw, ["痛经"]))) if first_col(raw, ["痛经"]) else np.nan,
            }
            brums = [c for c in raw.columns if str(c).startswith("6、")]
            brums_values = [BRUMS_MAP.get(text(r.get(c)), np.nan) for c in brums]
            row["brums_item_count"] = len(brums)
            row["brums_valid_item_count"] = int(pd.Series(brums_values).notna().sum())
            row["brums_total_score"] = float(np.nansum(brums_values)) if row["brums_valid_item_count"] else np.nan
            rows.append(row)
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw
    raw["player_in_baseline"] = raw["player_name"].isin(players)
    raw["morning_duplicate_response"] = raw.duplicated(["player_name", "date"], keep=False).astype(int)
    numeric_cols = [
        "sleep_quality_1_7", "morning_fatigue_1_7", "muscle_soreness_1_7", "mental_stress_1_7",
        "menstrual_discomfort", "brums_total_score",
    ]
    agg = raw.groupby(["player_name", "date"], dropna=False).agg(
        morning_response_count=("source_file", "size"),
        **{f"{c}_mean": (c, "mean") for c in numeric_cols},
        menstrual_flag=("menstrual_flag", lambda s: "；".join(sorted(set(x for x in s if x)))),
        brums_valid_item_count=("brums_valid_item_count", "max"),
    ).reset_index()
    agg["morning_reported"] = 1
    raw.to_csv(OUT / "morning_response_audit.csv", index=False, encoding="utf-8-sig")
    agg.to_csv(OUT / "morning_daily.csv", index=False, encoding="utf-8-sig")
    return agg


def audit_post_training(players: set[str]) -> pd.DataFrame:
    rows = []
    for path in sorted((SUBJECTIVE / "训练后").rglob("*.xlsx")):
        raw = pd.read_excel(path)
        name_col = first_col(raw, ["姓名与球衣号"]) or first_col(raw, ["姓名"])
        if not name_col:
            continue
        duration_col = first_col(raw, ["活动总时长"])
        rpe_col = first_col(raw, ["有多累"])
        activity_col = first_col(raw, ["活动类型"])
        date = file_date(path)
        for _, r in raw.iterrows():
            duration = numeric(r.get(duration_col)) if duration_col else np.nan
            rpe = numeric(r.get(rpe_col)) if rpe_col else np.nan
            rows.append(
                {
                    "player_name": name_key(r.get(name_col)),
                    "date": date,
                    "source_file": str(path),
                    "activity_type": text(r.get(activity_col)) if activity_col else "",
                    "duration_min": duration,
                    "rpe": rpe,
                    "srpe": duration * rpe if pd.notna(duration) and pd.notna(rpe) else np.nan,
                    "player_in_baseline": name_key(r.get(name_col)) in players,
                }
            )
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw
    raw.to_csv(OUT / "post_training_response_audit.csv", index=False, encoding="utf-8-sig")
    agg = raw.groupby(["player_name", "date"], dropna=False).agg(
        post_training_response_count=("source_file", "size"),
        post_training_session_count=("srpe", lambda s: int(s.notna().sum())),
        duration_min_sum=("duration_min", "sum"),
        rpe_mean=("rpe", "mean"),
        srpe_sum=("srpe", "sum"),
        activity_types=("activity_type", lambda s: "；".join(sorted(set(x for x in s if x)))),
    ).reset_index()
    agg["post_training_reported"] = 1
    agg.to_csv(OUT / "post_training_daily.csv", index=False, encoding="utf-8-sig")
    return agg


def build_injury(players: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_excel(INJURY_FILE)
    raw["player_name"] = raw["球员姓名"].map(name_key)
    raw["date"] = raw["受伤日期"].map(date_value)
    raw["player_in_baseline"] = raw["player_name"].isin(players)
    raw["athlete_day_key"] = raw["player_name"] + "|" + raw["date"].dt.strftime("%Y-%m-%d")
    daily = raw.groupby(["player_name", "date"], dropna=False).agg(
        injury_row_count=("athlete_day_key", "size"),
        injury_sites=("受伤部位", lambda s: "；".join(sorted(set(text(x) for x in s if text(x))))),
        injury_types=("损伤类型", lambda s: "；".join(sorted(set(text(x) for x in s if text(x))))),
        severity_text=("严重程度", lambda s: "；".join(sorted(set(text(x) for x in s if text(x))))),
        functional_impact=("功能影响", lambda s: "；".join(sorted(set(text(x) for x in s if text(x))))),
        pain_text=("疼痛等级", lambda s: "；".join(sorted(set(text(x) for x in s if text(x))))),
    ).reset_index()
    daily["injury"] = 1
    daily["injury_same_day_duplicate_rows"] = (daily["injury_row_count"] > 1).astype(int)
    raw.to_csv(OUT / "injury_row_audit.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(OUT / "injury_daily_label_audit.csv", index=False, encoding="utf-8-sig")
    return raw, daily


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    baseline, players = read_baseline()
    baseline.to_csv(OUT / "baseline_roster.csv", index=False, encoding="utf-8-sig")
    objective_files, objective_sessions = audit_objective(players)
    morning = audit_morning(players)
    post = audit_post_training(players)
    injury_raw, injury_daily = build_injury(players)

    dates = []
    for df in [objective_sessions, morning, post, injury_daily]:
        if not df.empty and "date" in df.columns:
            dates.extend(pd.to_datetime(df["date"], errors="coerce").dropna().tolist())
    min_date, max_date = min(dates), max(dates)
    calendar = pd.MultiIndex.from_product(
        [sorted(players), pd.date_range(min_date, max_date, freq="D")],
        names=["player_name", "date"],
    ).to_frame(index=False)
    selected_objective = objective_sessions[objective_sessions["selected_for_daily_aggregation"] == 1].copy()
    calendar = calendar.merge(
        selected_objective.groupby(["player_name", "date"], dropna=False).agg(
            objective_file_count_selected=("source_file", "size"),
            objective_raw_rows=("raw_rows", "sum"),
            objective_valid_time_rows=("valid_time_rows", "sum"),
            objective_session_duration_min=("session_duration_min", "sum"),
            speed_mean_ms=("speed_mean_ms", "mean"),
            speed_max_ms=("speed_max_ms", "max"),
            speed_valid_rate=("speed_valid_rate", "mean"),
            hr_mean_bpm=("hr_mean_bpm", "mean"),
            hr_max_bpm=("hr_max_bpm", "max"),
            satellite_mean=("satellite_mean", "mean"),
            signal_quality_mean=("signal_quality_mean", "mean"),
            hdop_mean=("hdop_mean", "mean"),
            objective_duplicate_day=("objective_duplicate_day", "max"),
            exact_binary_duplicate=("exact_binary_duplicate", "max"),
            objective_day_resolution=("objective_day_resolution", lambda s: "；".join(sorted(set(s)))),
        ).reset_index(),
        on=["player_name", "date"], how="left",
    )
    raw_file_counts = objective_sessions.groupby(["player_name", "date"], dropna=False).size().rename("objective_file_count_raw").reset_index()
    calendar = calendar.merge(raw_file_counts, on=["player_name", "date"], how="left")
    calendar = calendar.merge(morning, on=["player_name", "date"], how="left")
    calendar = calendar.merge(post, on=["player_name", "date"], how="left")
    calendar = calendar.merge(injury_daily, on=["player_name", "date"], how="left")
    calendar["injury"] = calendar["injury"].fillna(0).astype(int)
    calendar["injury_row_count"] = calendar["injury_row_count"].fillna(0).astype(int)
    calendar["objective_reported"] = calendar["objective_file_count_selected"].notna().astype(int)
    calendar["objective_file_count_selected"] = calendar["objective_file_count_selected"].fillna(0).astype(int)
    calendar["objective_file_count_raw"] = calendar["objective_file_count_raw"].fillna(0).astype(int)
    calendar["injury_only_day"] = ((calendar["injury"] == 1) & (calendar["objective_reported"] == 0) & calendar["morning_reported"].isna() & calendar["post_training_reported"].isna()).astype(int)
    calendar["monitoring_day"] = ((calendar["objective_reported"] == 1) | calendar["morning_reported"].notna() | calendar["post_training_reported"].notna()).astype(int)
    calendar["analysis_candidate_day"] = calendar["monitoring_day"]
    calendar["player_in_baseline"] = calendar["player_name"].isin(players)
    calendar = calendar.sort_values(["player_name", "date"]).reset_index(drop=True)
    calendar.to_csv(OUT / "target_domain_athlete_day_panel.csv", index=False, encoding="utf-8-sig")

    report = {
        "input": {
            "baseline": str(BASELINE_FILE),
            "injury": str(INJURY_FILE),
            "objective_root": str(RAW_OBJECTIVE),
            "subjective_root": str(SUBJECTIVE),
        },
        "analysis_unit": "athlete-calendar-day",
        "target_label": "injury = 1 when at least one physician injury record exists for that athlete on that calendar day; otherwise 0 within the reconstructed panel",
        "baseline_players": int(len(players)),
        "calendar_date_min": str(pd.Timestamp(min_date).date()),
        "calendar_date_max": str(pd.Timestamp(max_date).date()),
        "panel_rows": int(len(calendar)),
        "monitoring_days": int(calendar["monitoring_day"].sum()),
        "injury_positive_days": int(calendar["injury"].sum()),
        "injury_only_days": int(calendar["injury_only_day"].sum()),
        "objective_file_count": int(len(objective_files)),
        "objective_content_date_mismatches": int((objective_files.get("folder_content_date_match", pd.Series(dtype=bool)) == False).sum()) if not objective_files.empty else 0,
        "objective_unique_file_days_by_content_date": int(objective_sessions[["player_name", "date"]].drop_duplicates().shape[0]) if not objective_sessions.empty else 0,
        "objective_duplicate_days_by_content_date": int((objective_sessions.groupby(["player_name", "date"]).size() > 1).sum()) if not objective_sessions.empty else 0,
        "objective_selected_files_for_daily_aggregation": int(objective_sessions["selected_for_daily_aggregation"].sum()) if not objective_sessions.empty else 0,
        "morning_rows": int(len(morning)),
        "post_training_rows": int(len(post)),
        "injury_raw_rows": int(len(injury_raw)),
        "injury_unique_athlete_days": int(len(injury_daily)),
        "future_window_labels_created": False,
        "missing_reports_recoded_as_zero": False,
        "same_day_injury_rows_deleted": False,
        "next_review": [
            "Manual review of content-date mismatched objective files",
            "Confirm whether exact duplicate objective files should retain one canonical file",
            "Define functional-restriction sensitivity outcome from physician fields",
            "Rebuild historical features only after the objective-file resolution is approved",
        ],
    }
    (OUT / "daily_rebuild_label_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
