#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第四章第一阶段：将本地目标域重建为源域候选模型可审查的输入。

本程序不重新选择源域模型，也不使用本地结果反向调参。它只完成：
1) 读取已审计的本地运动员—日面板；
2) 将可比变量映射到源域8日模型的14个基础字段；
3) 用与源域相同的自然日历史窗口生成75个输入特征；
4) 对冻结的源域8日LightGBM进行一次直接外部验证。

结局保留本地队医记录的同日 injury 标签。缺失报告不改写为0，
没有可靠语义对应的变量保留缺失，并在映射审计表中明确标记。
"""

from __future__ import annotations

import hashlib
import os
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


BASE = Path(os.environ.get("LOCAL_DATA_ROOT", Path(__file__).resolve().parents[1]))
PANEL = BASE / "14_目标域日级重建/target_domain_athlete_day_panel.csv"
OBJECTIVE_AUDIT = BASE / "14_目标域日级重建/objective_session_audit.csv"
OUT = BASE / "15_源域候选迁移对齐"
SOURCE_ROOT = Path(os.environ.get("SOCCERMON_ROOT", Path(__file__).resolve().parents[2] / "source"))
SOURCE_HISTORY = SOURCE_ROOT / "outputs/injury_history_lopo_v1/history_8d.parquet"
FROZEN = SOURCE_ROOT / "outputs/injury_history_lopo_v1/frozen_source_candidate/SOCCERMON_INJURY_8D_LIGHTGBM_PIPELINE.joblib"
FEATURES_JSON = SOURCE_ROOT / "outputs/injury_history_lopo_v1/frozen_source_candidate/SELECTED_FEATURES.json"

FIELDS = [
    "session_rpe_mean",
    "session_srpe_sum_au",
    "session_duration_sum_min",
    "fatigue_burden",
    "poor_sleep_burden",
    "soreness_burden",
    "stress_burden",
    "sleep_duration_raw",
    "total_distance_m",
    "distance_above_15p0_km_h_m",
    "gps_max_speed_m_s",
    "mood_raw",
    "readiness_raw",
    "daily_load_au",
]
FLAGS = ["has_training_record", "has_wellness_record", "has_gps_record"]
WINDOW = 8


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def numeric(series):
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def gps_daily() -> tuple[pd.DataFrame, dict]:
    audit = pd.read_csv(OBJECTIVE_AUDIT)
    audit = audit.loc[audit["selected_for_daily_aggregation"].eq(1)].copy()
    rows = []
    for i, rec in audit.iterrows():
        path = Path(rec["source_file"])
        try:
            raw = pd.read_excel(path, usecols=["时间", "速度 m/s"])
            raw["timestamp"] = pd.to_datetime(raw["时间"], errors="coerce")
            raw["speed_m_s"] = numeric(raw["速度 m/s"])
            raw = raw.dropna(subset=["timestamp"]).sort_values("timestamp")
            raw["dt_s"] = raw["timestamp"].shift(-1).sub(raw["timestamp"]).dt.total_seconds()
            valid_dt = raw["dt_s"].gt(0) & raw["dt_s"].le(10)
            valid_speed = raw["speed_m_s"].ge(0)
            use = valid_dt & valid_speed
            speed = raw.loc[valid_speed, "speed_m_s"]
            rows.append({
                "player_name": rec["player_name"],
                "date": pd.to_datetime(rec["date"]).normalize(),
                "total_distance_m": float((raw.loc[use, "speed_m_s"] * raw.loc[use, "dt_s"]).sum()),
                "distance_above_15p0_km_h_m": float((raw.loc[use & raw["speed_m_s"].ge(15 / 3.6), "speed_m_s"] * raw.loc[use & raw["speed_m_s"].ge(15 / 3.6), "dt_s"]).sum()),
                "gps_max_speed_m_s": float(speed.max()) if speed.notna().any() else np.nan,
                "gps_valid_speed_rate": float(valid_speed.mean()) if len(raw) else np.nan,
                "gps_valid_interval_rate": float(use.mean()) if len(raw) else np.nan,
            })
        except Exception as exc:
            rows.append({"player_name": rec["player_name"], "date": pd.to_datetime(rec["date"]).normalize(), "read_error": str(exc)})
    daily = pd.DataFrame(rows)
    if daily.empty:
        raise ValueError("No local GPS files were available after the audit resolution")
    return daily, {"selected_files": int(len(audit)), "daily_rows": int(len(daily)), "read_errors": int(daily.get("read_error", pd.Series(dtype=object)).notna().sum())}


def make_raw_panel() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    panel = pd.read_csv(PANEL)
    panel.columns = [str(c).lstrip("\ufeff") for c in panel.columns]
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    gps, gps_audit = gps_daily()
    out = panel[["player_name", "date", "injury", "monitoring_day", "analysis_candidate_day"]].copy()
    out["session_rpe_mean"] = numeric(panel["rpe_mean"])
    out["session_srpe_sum_au"] = numeric(panel["srpe_sum"])
    out["session_duration_sum_min"] = numeric(panel["duration_min_sum"])
    for local, target in [
        ("morning_fatigue_1_7_mean", "fatigue_burden"),
        ("sleep_quality_1_7_mean", "poor_sleep_burden"),
        ("muscle_soreness_1_7_mean", "soreness_burden"),
        ("mental_stress_1_7_mean", "stress_burden"),
    ]:
        # Hooper item scale: 1 = best, 7 = worst; transform to source-like 0-1 burden.
        out[target] = (numeric(panel[local]) - 1.0) / 6.0
    # These fields have no verified semantic one-to-one correspondence in the current panel.
    out["sleep_duration_raw"] = np.nan
    out["mood_raw"] = np.nan
    out["readiness_raw"] = np.nan
    # Local sRPE sum is retained as a provisional daily-load proxy; it is flagged below.
    out["daily_load_au"] = numeric(panel["srpe_sum"])
    out["has_training_record"] = numeric(panel["post_training_reported"]).fillna(0)
    out["has_wellness_record"] = numeric(panel["morning_reported"]).fillna(0)
    out["has_gps_record"] = numeric(panel["objective_reported"]).fillna(0)
    out = out.merge(gps, on=["player_name", "date"], how="left", validate="one_to_one")
    out["injury"] = numeric(out["injury"]).fillna(0).astype(int)
    out.to_csv(OUT / "target_domain_raw_aligned_panel.csv", index=False, encoding="utf-8-sig")
    mapping = pd.DataFrame([
        ["session_rpe_mean", "rpe_mean", "direct", "训练后问卷RPE日均值"],
        ["session_srpe_sum_au", "srpe_sum", "direct", "训练后问卷sRPE日负荷"],
        ["session_duration_sum_min", "duration_min_sum", "direct", "训练后问卷训练时长日汇总"],
        ["fatigue_burden", "morning_fatigue_1_7_mean", "harmonized", "Hooper 1-7转换为(x-1)/6"],
        ["poor_sleep_burden", "sleep_quality_1_7_mean", "harmonized", "Hooper 1-7转换为(x-1)/6"],
        ["soreness_burden", "muscle_soreness_1_7_mean", "harmonized", "Hooper 1-7转换为(x-1)/6"],
        ["stress_burden", "mental_stress_1_7_mean", "harmonized", "Hooper 1-7转换为(x-1)/6"],
        ["sleep_duration_raw", "无当前字段", "unavailable", "保留缺失，不用睡眠质量替代睡眠时长"],
        ["total_distance_m", "GPS速度×有效时间间隔", "derived", "由原始GNSS速度和时间戳日级积分"],
        ["distance_above_15p0_km_h_m", "GPS速度≥15 km/h", "derived", "由原始GNSS速度和时间戳日级积分"],
        ["gps_max_speed_m_s", "GPS速度 m/s", "direct", "原始GNSS日级最大速度"],
        ["mood_raw", "无当前字段", "unavailable", "BRUMS总分不直接替代源域mood_raw"],
        ["readiness_raw", "无当前字段", "unavailable", "Hooper总分不直接替代源域readiness_raw"],
        ["daily_load_au", "srpe_sum", "proxy", "暂以本地sRPE日负荷作为训练负荷代理，需敏感性分析"],
        ["has_training_record", "post_training_reported", "availability", "训练后问卷是否有记录"],
        ["has_wellness_record", "morning_reported", "availability", "晨间问卷是否有记录"],
        ["has_gps_record", "objective_reported", "availability", "GNSS日文件是否有记录"],
    ], columns=["source_field", "local_field_or_rule", "status", "interpretation"])
    mapping.to_csv(OUT / "source_target_feature_mapping_audit.csv", index=False, encoding="utf-8-sig")
    return out, gps_audit, mapping


def make_history(raw: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    raw = raw.sort_values(["player_name", "date"])
    for player, g in raw.groupby("player_name", sort=True):
        calendar = pd.date_range(g.date.min(), g.date.max(), freq="D")
        daily = g.set_index("date").reindex(calendar)
        values = daily[FIELDS].apply(pd.to_numeric).replace([np.inf, -np.inf], np.nan)
        past = values.shift(1)
        observed = values.notna().any(axis=1).astype(float).shift(1).fillna(0)
        frame = pd.DataFrame(index=calendar)
        for field in FIELDS:
            r = past[field].rolling(WINDOW, min_periods=1)
            frame[field + "__mean"] = r.mean()
            frame[field + "__std"] = r.std(ddof=0)
            frame[field + "__max"] = r.max()
            frame[field + "__coverage"] = r.count() / WINDOW
            frame[field + "__previous_day"] = past[field]
        for flag in FLAGS:
            frame[flag + "__rate"] = daily[flag].fillna(0).shift(1).rolling(WINDOW, min_periods=1).sum() / WINDOW
        frame["history_observed_days"] = observed.rolling(WINDOW, min_periods=1).sum()
        frame["history_calendar_coverage"] = np.minimum(np.arange(len(calendar)), WINDOW) / WINDOW
        frame = frame.reindex(pd.DatetimeIndex(g.date))
        frame["player_name"] = player
        frame["date"] = frame.index
        frame["injury"] = g["injury"].to_numpy()
        frame["monitoring_day"] = g["monitoring_day"].to_numpy()
        frame["eligible_8d_history"] = frame["history_observed_days"].ge(1).astype(int).to_numpy()
        pieces.append(frame.reset_index(drop=True))
    result = pd.concat(pieces, ignore_index=True)
    result = result.loc[result["monitoring_day"].eq(1) & result["eligible_8d_history"].eq(1)].copy()
    result.to_csv(OUT / "target_domain_history_8d_aligned.csv", index=False, encoding="utf-8-sig")
    return result


def score_and_report(history: pd.DataFrame, mapping: pd.DataFrame) -> dict:
    selected = json.loads(FEATURES_JSON.read_text(encoding="utf-8"))["features"]
    missing = [c for c in selected if c not in history.columns]
    if missing:
        raise AssertionError(f"Target history is missing frozen feature columns: {missing}")
    pipe = joblib.load(FROZEN)
    score = pipe.predict_proba(history[selected].astype(float))[:, 1]
    out = history[["player_name", "date", "injury", "monitoring_day"]].copy()
    out["source_8d_lightgbm_score"] = score
    out.to_csv(OUT / "target_external_validation_predictions.csv", index=False, encoding="utf-8-sig")
    y = out["injury"].to_numpy(dtype=int)
    report = {
        "model": "frozen source 8d LightGBM",
        "source_model_file": str(FROZEN),
        "target_rows": int(len(out)),
        "target_players": int(out["player_name"].nunique()),
        "target_positive_days": int(y.sum()),
        "target_prevalence": float(y.mean()),
        "source_label_semantics": "athlete-submitted injury notification",
        "target_label_semantics": "physician-recorded same-day injury record",
        "direct_external_validation": {
            "AP": float(average_precision_score(y, score)) if y.sum() else None,
            "ROC_AUC": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None,
            "Brier_raw_output": float(brier_score_loss(y, score)),
            "LogLoss_raw_output": float(log_loss(y, np.clip(score, 1e-7, 1 - 1e-7), labels=[0, 1])),
            "mean_score": float(score.mean()),
        },
        "interpretation": "External validation only; no target-domain model selection or recalibration has been applied.",
        "mapping_proxy_fields": mapping.loc[mapping.status.eq("proxy"), "source_field"].tolist(),
        "unavailable_fields": mapping.loc[mapping.status.eq("unavailable"), "source_field"].tolist(),
    }
    (OUT / "target_external_validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw, gps_audit, mapping = make_raw_panel()
    history = make_history(raw)
    report = score_and_report(history, mapping)
    manifest = {
        "status": "COMPLETE_TARGET_ALIGNMENT_AND_DIRECT_EXTERNAL_VALIDATION",
        "script": str(Path(__file__).resolve()),
        "panel_sha256": sha256(PANEL),
        "objective_audit_sha256": sha256(OBJECTIVE_AUDIT),
        "source_history_sha256": sha256(SOURCE_HISTORY),
        "frozen_model": str(FROZEN),
        "window_days": WINDOW,
        "calendar_interval": "[date-8, date-1]",
        "target": "same-day physician-recorded injury label; no future label construction",
        "gps_audit": gps_audit,
        "report": report,
    }
    (OUT / "ALIGNMENT_RUN_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
