#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第四章目标域原始资料审计

目的：以附件中的当前原始资料为唯一输入，审计本地女足目标域的
运动员、日期、伤病记录、晨间主观报告、训练后负荷报告和GNSS/IMU文件。

本程序只做资料审计，不构造未来7天标签，不删除同日多伤病记录，
不把缺失报告改写为零，也不把审计结果直接当作最终模型数据。
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(os.environ.get("LOCAL_DATA_ROOT", Path(__file__).resolve().parents[1]))
RAW_OBJECTIVE = BASE / "00_原始客观数据"
SUBJECTIVE = BASE / "主观数据"
INJURY_FILE = BASE / "女足损伤记录表.xlsx"
BASELINE_FILE = BASE / "东北师范大学女足基线健康与既往损伤史调查问卷xlsx.xlsx"
OUT = BASE / "13_目标域原始审计"


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", "", str(value)).replace("　", "")


def normalize_name(value) -> str:
    """统一姓名键；保留原始姓名另存，不在此处做近似匹配。"""
    text = clean_text(value)
    text = re.sub(r"\(\d+\)$", "", text)
    text = re.sub(r"\s*\d{1,2}\.\d{2}\.\d{2}$", "", text)
    text = re.sub(r"^\d{1,2}号[-—_]", "", text)
    text = re.sub(r"^\d{1,2}[-—_]", "", text)
    return text


def parse_date(value) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT
    parsed = pd.to_datetime(str(value).strip(), errors="coerce")
    return parsed.normalize() if not pd.isna(parsed) else pd.NaT


def parse_folder_date(path: Path) -> pd.Timestamp:
    """从 6月/6.10 形式的目录解析日期。"""
    month_match = re.search(r"(\d{1,2})月", path.parent.name)
    day_match = re.match(r"^(\d{1,2})\.(\d{1,2})", path.name)
    if not month_match or not day_match:
        return pd.NaT
    return pd.Timestamp(2026, int(month_match.group(1)), int(day_match.group(2)))


def parse_subjective_file_date(path: Path) -> pd.Timestamp:
    month_match = re.search(r"(\d{1,2})月", path.parent.name)
    day_match = re.match(r"^(\d{1,2})\.(\d{1,2})", path.stem)
    if not month_match or not day_match:
        return pd.NaT
    return pd.Timestamp(2026, int(month_match.group(1)), int(day_match.group(2)))


def parse_numeric(value):
    if pd.isna(value):
        return np.nan
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else np.nan


def first_column(df: pd.DataFrame, contains: list[str]) -> str | None:
    for col in df.columns:
        text = str(col)
        if all(token in text for token in contains):
            return col
    return None


def load_baseline() -> tuple[pd.DataFrame, set[str]]:
    df = pd.read_excel(BASELINE_FILE)
    id_col = first_column(df, ["姓名", "设备编号"])
    if id_col is None:
        raise ValueError("基线问卷中未找到姓名与设备编号字段")
    df["raw_player_id"] = df[id_col].astype(str)
    df["player_name"] = df["raw_player_id"].map(normalize_name)
    return df, set(df["player_name"].dropna())


def audit_injury(players: set[str]) -> tuple[pd.DataFrame, dict]:
    df = pd.read_excel(INJURY_FILE)
    df["raw_player_name"] = df["球员姓名"].astype(str)
    df["player_name"] = df["球员姓名"].map(normalize_name)
    df["injury_date"] = df["受伤日期"].map(parse_date)
    df["athlete_day_key"] = df["player_name"] + "|" + df["injury_date"].dt.strftime("%Y-%m-%d")

    dup = df[df.duplicated("athlete_day_key", keep=False)].sort_values("athlete_day_key")
    day = (
        df.groupby(["player_name", "injury_date"], dropna=False)
        .agg(
            injury_rows=("athlete_day_key", "size"),
            injury_sites=("受伤部位", lambda s: "；".join(sorted(set(map(str, s))))),
            injury_types=("损伤类型", lambda s: "；".join(sorted(set(map(str, s))))),
            severity_values=("严重程度", lambda s: "；".join(sorted(set(map(str, s))))),
            functional_impact=("功能影响", lambda s: "；".join(sorted(set(map(str, s))))),
        )
        .reset_index()
    )
    day["injury_any_same_day"] = 1
    day["player_in_baseline"] = day["player_name"].isin(players)
    day.to_csv(OUT / "injury_athlete_day_audit.csv", index=False, encoding="utf-8-sig")
    dup.to_csv(OUT / "injury_same_day_duplicate_rows.csv", index=False, encoding="utf-8-sig")

    summary = {
        "raw_injury_rows": int(len(df)),
        "unique_athlete_days": int(len(day)),
        "duplicate_same_athlete_day_rows": int(len(dup)),
        "duplicate_athlete_day_keys": int(dup["athlete_day_key"].nunique()),
        "players_with_injury_rows": int(df["player_name"].nunique()),
        "players_not_in_baseline": sorted(set(df["player_name"]) - players),
        "date_min": str(df["injury_date"].min().date()),
        "date_max": str(df["injury_date"].max().date()),
        "functional_impact_counts": {str(k): int(v) for k, v in df["功能影响"].fillna("<NA>").value_counts().items()},
        "severity_counts": {str(k): int(v) for k, v in df["严重程度"].fillna("<NA>").value_counts().items()},
    }
    return df, summary


def audit_subjective(kind: str, players: set[str]) -> tuple[pd.DataFrame, dict]:
    root = SUBJECTIVE / kind
    files = sorted(root.rglob("*.xlsx"))
    rows = []
    for path in files:
        try:
            raw = pd.read_excel(path)
        except Exception as exc:
            rows.append({"source_file": str(path), "read_error": str(exc)})
            continue
        name_col = first_column(raw, ["姓名"])
        if name_col is None:
            name_col = first_column(raw, ["姓名与球衣号"])
        if name_col is None:
            continue
        date_col = first_column(raw, ["填写日期"]) if kind == "晨间" else None
        for _, record in raw.iterrows():
            row = {
                "source_file": str(path),
                "source_file_name": path.name,
                "raw_player_name": record.get(name_col, ""),
                "player_name": normalize_name(record.get(name_col, "")),
                "date": parse_date(record.get(date_col)) if date_col else parse_subjective_file_date(path),
            }
            for col in raw.columns:
                row[str(col)] = record.get(col)
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        summary = {"source_files": len(files), "rows": 0}
        return df, summary
    df["player_in_baseline"] = df["player_name"].isin(players)
    df.to_csv(OUT / f"{kind}_row_audit.csv", index=False, encoding="utf-8-sig")

    if kind == "晨间":
        numeric_cols = [c for c in df.columns if any(key in c for key in ["睡眠质量", "疲劳感", "肌肉酸痛", "心理压力"])]
        brums_cols = [c for c in df.columns if c.startswith("6、")]
        summary = {
            "source_files": len(files),
            "rows": int(len(df)),
            "unique_players": int(df["player_name"].nunique()),
            "unique_dates": int(df["date"].nunique()),
            "date_min": str(df["date"].min().date()),
            "date_max": str(df["date"].max().date()),
            "players_not_in_baseline": sorted(set(df["player_name"]) - players),
            "wellness_columns": numeric_cols,
            "wellness_nonmissing": {c: int(df[c].notna().sum()) for c in numeric_cols},
            "brums_column_count": len(brums_cols),
            "brums_columns": brums_cols,
        }
    else:
        duration_col = first_column(df, ["活动总时长"])
        rpe_col = first_column(df, ["有多累"])
        durations = df[duration_col].map(parse_numeric) if duration_col else pd.Series(dtype=float)
        rpes = df[rpe_col].map(parse_numeric) if rpe_col else pd.Series(dtype=float)
        summary = {
            "source_files": len(files),
            "rows": int(len(df)),
            "unique_players": int(df["player_name"].nunique()),
            "unique_dates": int(df["date"].nunique()),
            "date_min": str(df["date"].min().date()),
            "date_max": str(df["date"].max().date()),
            "players_not_in_baseline": sorted(set(df["player_name"]) - players),
            "duration_column": duration_col,
            "rpe_column": rpe_col,
            "duration_numeric": int(durations.notna().sum()),
            "rpe_numeric": int(rpes.notna().sum()),
            "srpe_numeric_pairs": int((durations.notna() & rpes.notna()).sum()),
        }
    return df, summary


def audit_objective(players: set[str]) -> tuple[pd.DataFrame, dict]:
    files = sorted(RAW_OBJECTIVE.rglob("*.xlsx"))
    records = []
    for path in files:
        date = parse_folder_date(path.parent)
        raw_stem = path.stem
        records.append(
            {
                "source_file": str(path),
                "raw_player_stem": raw_stem,
                "player_name": normalize_name(raw_stem),
                "date": date,
                "player_in_baseline": normalize_name(raw_stem) in players,
            }
        )
    df = pd.DataFrame(records)
    if df.empty:
        return df, {"source_files": 0}
    df["athlete_day_key"] = df["player_name"] + "|" + df["date"].dt.strftime("%Y-%m-%d")
    duplicates = df[df.duplicated("athlete_day_key", keep=False)].sort_values("athlete_day_key")
    df.to_csv(OUT / "objective_file_inventory.csv", index=False, encoding="utf-8-sig")
    duplicates.to_csv(OUT / "objective_duplicate_athlete_days.csv", index=False, encoding="utf-8-sig")
    summary = {
        "source_files": int(len(df)),
        "unique_players": int(df["player_name"].nunique()),
        "unique_dates": int(df["date"].nunique()),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "unique_athlete_days": int(df["athlete_day_key"].nunique()),
        "duplicate_file_rows": int(len(duplicates)),
        "duplicate_athlete_day_keys": int(duplicates["athlete_day_key"].nunique()),
        "players_not_in_baseline": sorted(set(df["player_name"]) - players),
    }
    return df, summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    baseline, players = load_baseline()
    baseline.to_csv(OUT / "baseline_roster_audit.csv", index=False, encoding="utf-8-sig")
    _, injury_summary = audit_injury(players)
    _, morning_summary = audit_subjective("晨间", players)
    _, post_summary = audit_subjective("训练后", players)
    _, objective_summary = audit_objective(players)

    report = {
        "input_files": {
            "baseline": str(BASELINE_FILE),
            "injury": str(INJURY_FILE),
            "objective_root": str(RAW_OBJECTIVE),
            "subjective_root": str(SUBJECTIVE),
        },
        "baseline": {"rows": int(len(baseline)), "unique_players": int(len(players))},
        "injury": injury_summary,
        "morning": morning_summary,
        "post_training": post_summary,
        "objective": objective_summary,
        "analysis_boundary": {
            "unit": "athlete-calendar-day",
            "injury_primary_candidate": "injury_any_same_day",
            "future_window_labels_created": False,
            "missing_reports_recoded_as_zero": False,
            "same_day_duplicate_injury_rows_deleted": False,
        },
    }
    (OUT / "target_domain_raw_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
