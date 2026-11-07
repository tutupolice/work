"""Nested, athlete-disjoint Platt calibration for the selected injury model."""
from pathlib import Path
import hashlib
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from threadpoolctl import threadpool_limits
from config_soccermon import make_pipeline, BASELINE_PARAMS

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/injury_history_lopo_v1'
WINDOW = 8
MODEL = 'LightGBM'

def athlete_splits(d):
    subjects = d.groupby('player_name', sort=True).injury.max()
    if subjects.sum() < 3 or (len(subjects) - subjects.sum()) < 3:
        raise ValueError('Not enough positive/negative athletes for 3-fold calibration.')
    for tr, va in StratifiedKFold(3, shuffle=True, random_state=42).split(subjects.index, subjects):
        train_players = set(subjects.index[tr]); valid_players = set(subjects.index[va])
        train = np.flatnonzero(d.player_name.isin(train_players).to_numpy())
        valid = np.flatnonzero(d.player_name.isin(valid_players).to_numpy())
        yield train, valid

def make_base():
    model = make_pipeline(MODEL, dict(BASELINE_PARAMS[MODEL]))
    est = model.named_steps['model']
    if 'n_jobs' in est.get_params(): est.set_params(n_jobs=2)
    return model

def raw_fit_predict(d, features, train, test):
    model = make_base()
    with warnings.catch_warnings(record=True) as ws, threadpool_limits(limits=2):
        model.fit(d.iloc[train][features], d.iloc[train].injury)
        pred = model.predict_proba(d.iloc[test][features])[:, 1]
    return pred, [str(x.message) for x in ws]

def calibrator_from_inner_oof(d, features, train):
    train_d = d.iloc[train].reset_index(drop=True)
    raw = np.full(len(train_d), np.nan)
    warnings_seen = []
    for inner_train, inner_valid in athlete_splits(train_d):
        pred, ws = raw_fit_predict(train_d, features, inner_train, inner_valid)
        raw[inner_valid] = pred
        warnings_seen.extend(ws)
    assert np.isfinite(raw).all()
    z = np.log(np.clip(raw, 1e-7, 1 - 1e-7) / np.clip(1 - raw, 1e-7, 1 - 1e-7)).reshape(-1, 1)
    cal = LogisticRegression(C=1e6, solver='lbfgs', max_iter=2000, random_state=42)
    cal.fit(z, train_d.injury.to_numpy())
    return cal, warnings_seen, float(average_precision_score(train_d.injury, raw))

def calibrated_predict(cal, raw):
    z = np.log(np.clip(raw, 1e-7, 1 - 1e-7) / np.clip(1 - raw, 1e-7, 1 - 1e-7)).reshape(-1, 1)
    return cal.predict_proba(z)[:, 1]

def metric_row(y, p, label):
    return dict(prediction=label, AP=float(average_precision_score(y, p)),
                ROC_AUC=float(roc_auc_score(y, p)), Brier=float(brier_score_loss(y, p)),
                LogLoss=float(log_loss(y, np.clip(p, 1e-7, 1-1e-7))), mean_probability=float(np.mean(p)))

def main():
    d = pd.read_parquet(OUT / f'history_{WINDOW}d.parquet')
    bundle = __import__('joblib').load(OUT / 'selected_injury_model.joblib')
    features = bundle['features']
    raw_all = []; cal_all = []; warnings_seen = []; fold_meta = []
    for fold, player in enumerate(sorted(d.player_name.unique())):
        test = np.flatnonzero(d.player_name.eq(player).to_numpy())
        train = np.flatnonzero(d.player_name.ne(player).to_numpy())
        cal, ws, inner_ap = calibrator_from_inner_oof(d, features, train)
        raw, ws2 = raw_fit_predict(d, features, train, test)
        calibrated = calibrated_predict(cal, raw)
        frame = d.iloc[test][['player_name', 'date', 'injury']].copy()
        frame['raw_probability'] = raw
        frame['calibrated_probability'] = calibrated
        raw_all.append(frame)
        warnings_seen.extend(ws + ws2)
        fold_meta.append({'fold': fold, 'test_player': player, 'inner_oof_raw_AP': inner_ap,
                          'calibration_intercept': float(cal.intercept_[0]),
                          'calibration_slope': float(cal.coef_[0, 0])})
        print(f'[calibration] LOPO {fold+1}/50', flush=True)
    pred = pd.concat(raw_all, ignore_index=True)
    pred.to_csv(OUT / 'selected_model_calibrated_oof_predictions.csv', index=False)
    y = pred.injury.to_numpy()
    metrics = [metric_row(y, pred.raw_probability, 'raw'), metric_row(y, pred.calibrated_probability, 'calibrated')]
    cal_stats = pd.DataFrame(fold_meta)
    cal_stats.to_csv(OUT / 'calibration_fold_statistics.csv', index=False)
    report = {'model': MODEL, 'window_days': WINDOW, 'rows': int(len(pred)), 'positives': int(y.sum()),
              'players': int(pred.player_name.nunique()), 'metrics': metrics,
              'calibration_intercept_mean': float(cal_stats.calibration_intercept.mean()),
              'calibration_slope_mean': float(cal_stats.calibration_slope.mean()),
              'calibration_method': 'Platt/logistic calibration fitted on 3 athlete-disjoint inner-fold OOF scores inside each outer training set',
              'not_used_for_calibration': 'outer test player',
              'warning_count': len(warnings_seen),
              'source_prediction_sha256': hashlib.sha256((OUT / 'selected_model_oof_predictions.csv').read_bytes()).hexdigest()}
    (OUT / 'calibration_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
