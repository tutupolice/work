#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Target-domain update audit with athlete-level cross-fitting.

Completed in this script:
  1. frozen source-model direct external validation;
  2. intercept-only target calibration, cross-fitted by athlete;
  3. intercept+slope target calibration, cross-fitted by athlete;
  4. fixed-hyperparameter target-domain LightGBM refit, cross-fitted by athlete;
  5. athlete-cluster bootstrap intervals for the resulting out-of-fold scores.

No target-domain model selection, resampling, threshold optimization or
feature selection is performed. The target refit keeps the frozen source
feature set and source LightGBM hyperparameters unchanged.
"""

from __future__ import annotations

import hashlib
import os
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


SOURCE_ROOT = Path(os.environ.get("SOCCERMON_ROOT", Path(__file__).resolve().parents[2] / "source"))
BASE = Path(os.environ.get("LOCAL_DATA_ROOT", Path(__file__).resolve().parents[1]))
ALIGN = BASE / '15_源域候选迁移对齐'
INPUT = ALIGN / 'target_domain_history_8d_aligned.csv'
MODEL_FILE = SOURCE_ROOT / 'outputs/injury_history_lopo_v1/frozen_source_candidate/SOCCERMON_INJURY_8D_LIGHTGBM_PIPELINE.joblib'
OUTDIR = BASE / '16_目标域模型更新_LOPO'
SEED = 42
BOOTSTRAPS = 2000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def clip_prob(p):
    return np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)


def logit(p):
    p = clip_prob(p)
    return np.log(p / (1 - p))


def sigmoid(z):
    z = np.clip(np.asarray(z, dtype=float), -35, 35)
    return 1 / (1 + np.exp(-z))


def fit_calibration(z, y, mode):
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    if mode == 'intercept':
        def objective(theta):
            p = sigmoid(z + theta[0])
            return log_loss(y, p, labels=[0, 1])
        res = minimize(objective, np.array([0.0]), method='BFGS')
        return float(res.x[0]), 1.0
    if mode == 'intercept_slope':
        def objective(theta):
            p = sigmoid(theta[0] + theta[1] * z)
            return log_loss(y, p, labels=[0, 1]) + 1e-4 * (theta[1] - 1.0) ** 2
        res = minimize(objective, np.array([0.0, 1.0]), method='BFGS')
        return float(res.x[0]), float(res.x[1])
    raise ValueError(mode)


def metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = clip_prob(p)
    out = {
        'n': int(len(y)),
        'positive_days': int(y.sum()),
        'prevalence': float(y.mean()),
        'AP': float(average_precision_score(y, p)),
        'ROC_AUC': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        'Brier': float(brier_score_loss(y, p)),
        'LogLoss': float(log_loss(y, p, labels=[0, 1])),
        'mean_score': float(np.mean(p)),
    }
    return out


def bootstrap_metrics(frame, pred_col, n_boot=BOOTSTRAPS, seed=SEED):
    rng = np.random.default_rng(seed)
    players = frame['player_name'].drop_duplicates().to_numpy()
    vals = []
    for _ in range(n_boot):
        sampled = rng.choice(players, size=len(players), replace=True)
        pieces = [frame.loc[frame['player_name'].eq(player)] for player in sampled]
        b = pd.concat(pieces, ignore_index=True)
        if b['injury'].nunique() < 2:
            continue
        vals.append({
            'AP': average_precision_score(b['injury'], b[pred_col]),
            'ROC_AUC': roc_auc_score(b['injury'], b[pred_col]),
            'Brier': brier_score_loss(b['injury'], b[pred_col]),
            'LogLoss': log_loss(b['injury'], clip_prob(b[pred_col]), labels=[0, 1]),
        })
    boot = pd.DataFrame(vals)
    result = {}
    for col in ['AP', 'ROC_AUC', 'Brier', 'LogLoss']:
        result[col] = {
            'n_boot_valid': int(len(boot)),
            'q025': float(boot[col].quantile(0.025)),
            'median': float(boot[col].quantile(0.5)),
            'q975': float(boot[col].quantile(0.975)),
        }
    return result


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT)
    model = joblib.load(MODEL_FILE)
    feature_names = list(model.feature_names_in_)
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f'missing frozen features: {missing}')
    df = df.loc[df['monitoring_day'].eq(1) & df['eligible_8d_history'].eq(1)].copy()
    df['injury'] = df['injury'].astype(int)
    df['source_score'] = model.predict_proba(df[feature_names].astype(float))[:, 1]
    df['source_logit'] = logit(df['source_score'])

    players = np.sort(df['player_name'].unique())
    pred = {
        'source_direct': np.full(len(df), np.nan),
        'cal_intercept': np.full(len(df), np.nan),
        'cal_intercept_slope': np.full(len(df), np.nan),
        'target_fixed_lgbm': np.full(len(df), np.nan),
    }
    for heldout in players:
        test_mask = df['player_name'].eq(heldout).to_numpy()
        train_mask = ~test_mask
        y_train = df.loc[train_mask, 'injury'].to_numpy()
        z_train = df.loc[train_mask, 'source_logit'].to_numpy()
        z_test = df.loc[test_mask, 'source_logit'].to_numpy()
        a0, b0 = fit_calibration(z_train, y_train, 'intercept')
        a1, b1 = fit_calibration(z_train, y_train, 'intercept_slope')
        pred['source_direct'][test_mask] = df.loc[test_mask, 'source_score']
        pred['cal_intercept'][test_mask] = sigmoid(z_test + a0)
        pred['cal_intercept_slope'][test_mask] = sigmoid(a1 + b1 * z_test)

        # Fixed source architecture and hyperparameters; fit only on the 23
        # training athletes, with the source feature set and natural prevalence.
        source_lgbm = model.named_steps['model']
        params = source_lgbm.get_params()
        params.pop('class_weight', None)
        params.pop('n_jobs', None)
        params.update({'random_state': SEED, 'n_jobs': -1, 'verbosity': -1})
        # Reuse the source pipeline's frozen imputer so the target refit keeps
        # exactly the same 80-column source feature contract, including fields
        # that are unavailable in the local cohort.
        imp = model.named_steps['imputer']
        x_train = imp.transform(df.loc[train_mask, feature_names].astype(float))
        x_test = imp.transform(df.loc[test_mask, feature_names].astype(float))
        target_model = LGBMClassifier(**params)
        target_model.fit(x_train, y_train)
        pred['target_fixed_lgbm'][test_mask] = target_model.predict_proba(x_test)[:, 1]

    for col, values in pred.items():
        df[col] = values
    df.to_csv(OUTDIR / 'target_domain_lopo_update_predictions.csv', index=False, encoding='utf-8-sig')

    methods = {
        'frozen_source_direct': 'source_direct',
        'target_intercept_calibration': 'cal_intercept',
        'target_intercept_slope_calibration': 'cal_intercept_slope',
        'target_fixed_hyperparameter_lgbm_refit': 'target_fixed_lgbm',
    }
    summary = {}
    bootstrap = {}
    for method, col in methods.items():
        summary[method] = metrics(df['injury'], df[col])
        bootstrap[method] = bootstrap_metrics(df[['player_name', 'injury', col]].rename(columns={col: 'pred'}), 'pred')

    manifest = {
        'status': 'COMPLETE_TARGET_DOMAIN_LOPO_UPDATE_AUDIT',
        'script': str(Path(__file__)),
        'input': str(INPUT),
        'input_sha256': sha256(INPUT),
        'frozen_model': str(MODEL_FILE),
        'frozen_model_sha256': sha256(MODEL_FILE),
        'random_seed': SEED,
        'outer_split': 'athlete-level LOPO across target-domain athletes',
        'target_rows': int(len(df)),
        'target_players': int(df['player_name'].nunique()),
        'target_positive_days': int(df['injury'].sum()),
        'target_prevalence': float(df['injury'].mean()),
        'feature_count': len(feature_names),
        'feature_set': feature_names,
        'strategies': {
            'frozen_source_direct': 'no target label used',
            'target_intercept_calibration': 'athlete-LOPO intercept-only logistic calibration on source logit',
            'target_intercept_slope_calibration': 'athlete-LOPO logistic calibration on source logit with intercept and slope',
            'target_fixed_hyperparameter_lgbm_refit': 'athlete-LOPO target refit with frozen source LightGBM hyperparameters and feature set; no resampling or class weighting',
        },
        'summary': summary,
        'athlete_cluster_bootstrap': bootstrap,
        'limitations': [
            'Source and target injury labels have different recording mechanisms.',
            'Target domain has one proxy field and three unavailable source fields.',
            'No target-domain hyperparameter search, feature selection, resampling or threshold optimization was performed.',
            'Target-domain results are internal cross-fitted update estimates, not prospective external validation.',
        ],
    }
    (OUTDIR / 'target_domain_lopo_update_report.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: summary[k] for k in summary}, ensure_ascii=False, indent=2))
    print(OUTDIR)


if __name__ == '__main__':
    main()
