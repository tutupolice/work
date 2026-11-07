"""Export a full-source model with an inner-OOF Platt calibrator."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from config_soccermon import make_pipeline, BASELINE_PARAMS
import importlib

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/injury_history_lopo_v1'
cal = importlib.import_module('04c_injury_nested_probability_calibration')

def main():
    d = pd.read_parquet(OUT / 'history_8d.parquet')
    features = joblib.load(OUT / 'selected_injury_model.joblib')['features']
    indices = np.arange(len(d))
    calibrator, warnings_seen, inner_ap = cal.calibrator_from_inner_oof(d, features, indices)
    base = make_pipeline('LightGBM', dict(BASELINE_PARAMS['LightGBM']))
    base.fit(d[features], d.injury)
    bundle = {'base_pipeline': base, 'calibrator': calibrator, 'features': features,
              'model': 'LightGBM', 'window_days': 8,
              'calibration': 'Platt/logistic calibration on 3 athlete-disjoint inner OOF folds',
              'outcome': 'original same-day injury_report_t',
              'feature_timing': 'calendar history strictly before outcome date',
              'inner_oof_raw_AP': inner_ap,
              'warning_count': len(warnings_seen),
              'usage': 'raw = base_pipeline.predict_proba(X)[:,1]; calibrated = calibrator.predict_proba(logit(raw))[:,1]'}
    joblib.dump(bundle, OUT / 'selected_injury_model_calibrated.joblib')
    (OUT / 'calibrated_model_export.json').write_text(json.dumps({k:v for k,v in bundle.items() if k not in ['base_pipeline', 'calibrator']}, ensure_ascii=False, indent=2))
    print('saved', OUT / 'selected_injury_model_calibrated.joblib')

if __name__ == '__main__':
    main()
