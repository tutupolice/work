#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Freeze the current original-injury-label SoccerMon transfer candidate.

This is intentionally separate from the obsolete t+1/146-positive export
script.  It freezes the prespecified development-stage candidate:
8-day history + LightGBM + natural-prevalence training on the common cohort.
No local target-domain data are read here.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config_soccermon import make_pipeline


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/injury_history_lopo_v1/frozen_source_candidate"
HISTORY = ROOT / "outputs/injury_history_lopo_v1/history_8d.parquet"
BENCHMARK = ROOT / "outputs/injury_history_lopo_v1/benchmark_summary.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def native(value):
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(HISTORY)
    required = {"player_name", "date", "injury"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    features = [c for c in frame.columns if c not in {"player_name", "date", "injury"}]
    if len(frame) != 19613 or int(frame["injury"].sum()) != 149 or frame["player_name"].nunique() != 50:
        raise AssertionError("The 8-day history file does not match the locked current injury-label cohort")
    if not features or frame[features].select_dtypes(exclude=[np.number]).shape[1]:
        raise AssertionError("The frozen feature matrix must contain numeric predictors only")

    summary = pd.read_csv(BENCHMARK)
    selected = summary.loc[summary["configuration"].eq("8d_LightGBM_control")]
    if len(selected) != 1:
        raise AssertionError("8d_LightGBM_control is not uniquely present in the benchmark summary")

    x = frame[features].astype(float)
    y = frame["injury"].astype(int)
    pipeline = make_pipeline("LightGBM")
    pipeline.fit(x, y)

    joblib.dump(pipeline, OUT / "SOCCERMON_INJURY_8D_LIGHTGBM_PIPELINE.joblib", compress=3)
    (OUT / "SELECTED_FEATURES.json").write_text(
        json.dumps({"window_days": 8, "features": features}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    model_card = f"""# SoccerMon源域冻结迁移候选

## 冻结规则

依据源域开发阶段预先规定的AP优先规则，冻结 `8-day history + LightGBM + natural-prevalence control` 作为本地目标域迁移候选。该选择不使用本地女足数据，也不声称这是经过完整联合模型选择后稳定的最终冠军。

## 输入与结局

- 源域输入：{len(frame):,} 条运动员日、50 名运动员、{int(y.sum())} 个 `injury=1` 标签。
- 结局：原始 `injury_report_t` 复制为 `injury`，不做 t+1 平移。
- 历史窗口：日期 t 之前的 8 个自然日，即 [t-8, t-1]。
- 特征数：{len(features)}。
- 训练策略：自然类别比例；不使用 SMOTE 或随机过采样。
- 预处理：中位数插补；LightGBM 不做尺度标准化。
- 概率状态：原始模型输出，未完成目标域校准，不直接视为临床概率。

## 迁移使用边界

本模型必须先在本地目标域进行直接外部验证，再决定是否进行目标域概率校准或有限更新。不得根据本地测试结果重新挑选源域模型。
"""
    (OUT / "MODEL_CARD.md").write_text(model_card, encoding="utf-8")
    manifest = {
        "status": "COMPLETE_FROZEN_SOURCE_CANDIDATE",
        "script": str(Path(__file__).resolve()),
        "history_file": str(HISTORY),
        "history_sha256": sha256(HISTORY),
        "benchmark_file": str(BENCHMARK),
        "benchmark_sha256": sha256(BENCHMARK),
        "selection_rule": "development-stage AP priority; 8d_LightGBM_control",
        "model": "LightGBM",
        "window_days": 8,
        "target": "injury",
        "target_definition": "original athlete-submitted injury notification on the same calendar day; no label shift",
        "prediction_history": "calendar interval [date-8, date-1]",
        "rows": int(len(frame)),
        "positive_labels": int(y.sum()),
        "players": int(frame["player_name"].nunique()),
        "feature_count": len(features),
        "features": features,
        "preprocessing": ["median_imputation"],
        "balance_mode": "natural_prevalence_control",
        "probability_status": "raw_model_output_not_calibrated",
        "completed_utc": datetime.now(timezone.utc),
        "python": sys.version,
        "platform": platform.platform(),
    }
    (OUT / "MANIFEST.json").write_text(
        json.dumps(native(manifest), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[03s] COMPLETE | model=LightGBM | window=8d | rows={len(frame)} | positives={int(y.sum())}")
    print(f"[03s] pipeline={OUT / 'SOCCERMON_INJURY_8D_LIGHTGBM_PIPELINE.joblib'}")


if __name__ == "__main__":
    main()
