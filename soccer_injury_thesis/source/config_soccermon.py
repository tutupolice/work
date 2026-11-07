#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared configuration and estimator factory for the SoccerMon pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent
WINDOWS = (2, 4, 8, 16, 21, 32)
TARGET = "y_t1"
PLAYER = "player_name"
DATE = "date"
NEXT_DATE = "target_date"
RANDOM_STATE = 42
PRIMARY_METRIC = "average_precision"

FORBIDDEN_PREDICTORS = {
    "y_t1",
    "injury_report_t",
    "target_date",
    "outcome_boundary_flag",
    "monitoring_last_date",
    "pmsys_eligible",
    "calendar_year",
    "player_name",
    "date",
    "team",
}

OUTPUT_ROOT = ROOT / "outputs"
PREPROCESSING_DIR = OUTPUT_ROOT / "01_preprocessing"
FEATURE_DIR = OUTPUT_ROOT / "02_features"
BENCHMARK_DIR = OUTPUT_ROOT / "03a_benchmark"
TUNING_DIR = OUTPUT_ROOT / "03b_tuning"
EVALUATION_DIR = OUTPUT_ROOT / "04_evaluation"
INTERPRETATION_DIR = OUTPUT_ROOT / "05_interpretation"
FINAL_DIR = OUTPUT_ROOT / "06_final_model"

PANEL_PATH = ROOT / "verified_outputs/08_source_athlete_day_t1_v1/source_athlete_day_t1.parquet"
FIELD_MAPPING_PATH = ROOT / "verified_outputs/08_source_athlete_day_t1_v1/SOURCE_FIELD_MAPPING.csv"
MISSINGNESS_PATH = ROOT / "verified_outputs/08_source_athlete_day_t1_v1/SOURCE_MISSINGNESS.csv"

SOURCE_COMMON_CURRENT = [
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
]

SOURCE_NATIVE = [
    "daily_load_au",
    "weekly_load_au",
    "atl_exported_au",
    "ctl28_exported_au",
    "ctl42_exported_au",
    "acwr_exported",
    "monotony_exported",
    "strain_exported_au",
]

SOURCE_HISTORY = [
    "history_ever_prior_injury_report",
    "history_past_injury_report_days",
    "history_days_since_last_injury_report",
]

MODEL_NAMES = (
    "LogisticRegression",
    "GaussianNB",
    "SVM-RBF",
    "RandomForest",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "MLP",
)

MODEL_FAMILY = {
    "LogisticRegression": "linear",
    "GaussianNB": "probabilistic",
    "SVM-RBF": "kernel",
    "RandomForest": "bagging_tree",
    "XGBoost": "boosting_tree",
    "LightGBM": "boosting_tree",
    "CatBoost": "boosting_tree",
    "MLP": "neural_network",
}

BASELINE_PARAMS = {
    "LogisticRegression": {"C": 1.0, "penalty": "l2"},
    "GaussianNB": {"var_smoothing": 1e-9},
    "SVM-RBF": {"C": 1.0, "gamma": "scale"},
    "RandomForest": {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 5},
    "XGBoost": {
        "n_estimators": 180, "max_depth": 3, "learning_rate": 0.04,
        "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.0, "reg_lambda": 1.0,
    },
    "LightGBM": {
        "n_estimators": 180, "max_depth": 3, "num_leaves": 7,
        "learning_rate": 0.04, "min_child_samples": 40,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.0, "reg_lambda": 1.0,
    },
    "CatBoost": {"iterations": 180, "depth": 4, "learning_rate": 0.04, "l2_leaf_reg": 3.0},
    "MLP": {
        "hidden_layer_sizes": (32,), "alpha": 0.001,
        "learning_rate_init": 0.001, "max_iter": 150,
    },
}

TUNING_SPACES = {
    "LogisticRegression": {
        "model__C": np.logspace(-2, 1, 12),
        "model__penalty": ["l1", "l2"],
    },
    "GaussianNB": {"model__var_smoothing": np.logspace(-11, -7, 10)},
    "SVM-RBF": {
        "model__C": np.logspace(-1, 1, 10),
        "model__gamma": ["scale", "auto"],
    },
    "RandomForest": {
        "model__n_estimators": [150, 250],
        "model__max_depth": [4, 8, None],
        "model__min_samples_leaf": [2, 5, 10],
    },
    "XGBoost": {
        "model__n_estimators": [120, 180, 260],
        "model__max_depth": [2, 3, 4],
        "model__learning_rate": [0.02, 0.04, 0.08],
        "model__min_child_weight": [1, 5, 10],
        "model__subsample": [0.7, 0.8, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 1.0],
        "model__reg_alpha": [0.0, 0.1, 1.0],
        "model__reg_lambda": [1.0, 5.0, 10.0],
    },
    "LightGBM": {
        "model__n_estimators": [120, 180, 260],
        "model__max_depth": [2, 3, 4],
        "model__num_leaves": [3, 7, 15],
        "model__learning_rate": [0.02, 0.04, 0.08],
        "model__min_child_samples": [20, 40, 80],
        "model__subsample": [0.7, 0.8, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 1.0],
        "model__reg_alpha": [0.0, 0.1, 1.0],
        "model__reg_lambda": [1.0, 5.0, 10.0],
    },
    "CatBoost": {
        "model__iterations": [120, 180, 260],
        "model__depth": [3, 4, 5],
        "model__learning_rate": [0.02, 0.04, 0.08],
        "model__l2_leaf_reg": [1.0, 3.0, 10.0],
    },
    "MLP": {
        "model__hidden_layer_sizes": [(16,), (32,), (32, 16)],
        "model__alpha": [1e-4, 1e-3, 1e-2],
        "model__learning_rate_init": [3e-4, 1e-3, 3e-3],
        "model__max_iter": [120, 180],
    },
}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def ensure_output_dirs() -> None:
    for directory in (PREPROCESSING_DIR, FEATURE_DIR, BENCHMARK_DIR, TUNING_DIR,
                      EVALUATION_DIR, INTERPRETATION_DIR, FINAL_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def make_pipeline(model_name: str, params: dict[str, Any] | None = None) -> Pipeline:
    """Build the primary natural-prevalence estimator pipeline."""
    values = dict(BASELINE_PARAMS[model_name])
    if params:
        values.update(params)
    if model_name == "LogisticRegression":
        model = LogisticRegression(
            C=values["C"], penalty=values["penalty"], solver="liblinear",
            class_weight=None, max_iter=2000, random_state=RANDOM_STATE,
        )
        scale = True
    elif model_name == "GaussianNB":
        model = GaussianNB(var_smoothing=values["var_smoothing"])
        scale = False
    elif model_name == "SVM-RBF":
        model = SVC(
            C=values["C"], gamma=values["gamma"], kernel="rbf",
            probability=False, class_weight=None, cache_size=2048, random_state=RANDOM_STATE,
        )
        scale = True
    elif model_name == "RandomForest":
        model = RandomForestClassifier(
            n_estimators=values["n_estimators"], max_depth=values["max_depth"],
            min_samples_leaf=values["min_samples_leaf"], class_weight=None,
            n_jobs=-1, random_state=RANDOM_STATE,
        )
        scale = False
    elif model_name == "XGBoost":
        import xgboost as xgb
        model = xgb.XGBClassifier(
            objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
            n_jobs=-1, random_state=RANDOM_STATE, scale_pos_weight=1.0,
            **values,
        )
        scale = False
    elif model_name == "LightGBM":
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            objective="binary", verbosity=-1, deterministic=True,
            force_col_wise=True, n_jobs=-1, random_state=RANDOM_STATE,
            class_weight=None, **values,
        )
        scale = False
    elif model_name == "CatBoost":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(
            loss_function="Logloss", verbose=False, allow_writing_files=False,
            thread_count=-1, random_seed=RANDOM_STATE, auto_class_weights=None, **values,
        )
        scale = False
    elif model_name == "MLP":
        model = MLPClassifier(
            hidden_layer_sizes=values["hidden_layer_sizes"], alpha=values["alpha"],
            learning_rate_init=values["learning_rate_init"], max_iter=values["max_iter"],
            early_stopping=False, random_state=RANDOM_STATE, batch_size=256,
        )
        scale = True
    else:
        raise ValueError(f"Unknown model: {model_name}")
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median", add_indicator=False))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)
