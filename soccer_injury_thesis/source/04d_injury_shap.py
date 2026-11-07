"""SHAP explanation for the frozen, full-source injury candidate model."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/injury_history_lopo_v1'
FIG = OUT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)

def main():
    bundle = joblib.load(OUT / 'selected_injury_model.joblib')
    d = pd.read_parquet(OUT / 'history_8d.parquet')
    features = bundle['features']
    prep = bundle['preprocessor']; model = bundle['estimator']
    X = prep.transform(d[features])
    rng = np.random.default_rng(42)
    ids = np.sort(rng.choice(len(d), size=min(2500, len(d)), replace=False))
    Xs = X[ids]
    explain = shap.TreeExplainer(model)
    values = explain(Xs)
    shap_values = values.values
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]
    table = pd.DataFrame({'feature': features,
                          'mean_abs_shap': np.abs(shap_values).mean(axis=0),
                          'mean_shap': shap_values.mean(axis=0)})
    table = table.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    table['rank'] = np.arange(1, len(table) + 1)
    table.to_csv(OUT / 'shap_global_feature_contributions.csv', index=False)
    np.save(OUT / 'shap_values_sample.npy', shap_values)
    pd.DataFrame(Xs, columns=features).to_parquet(OUT / 'shap_feature_sample.parquet', index=False)
    pd.DataFrame({'row_index': ids, 'player_name': d.iloc[ids].player_name.to_numpy(),
                  'date': d.iloc[ids].date.to_numpy(), 'injury': d.iloc[ids].injury.to_numpy()}).to_csv(OUT / 'shap_sample_index.csv', index=False)

    top = table.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.0, 6.5))
    ax.barh(top.feature, top.mean_abs_shap, color='#245B8F')
    ax.set_xlabel('Mean absolute SHAP value (model contribution)')
    ax.set_ylabel('Feature')
    ax.set_title('Global predictive contributions in the frozen LightGBM model')
    ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(FIG / 'fig3_6_shap_global_importance.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG / 'fig3_6_shap_global_importance.svg', bbox_inches='tight')
    plt.close(fig)
    report = {'model': 'LightGBM', 'window_days': 8, 'rows_explained': int(len(ids)),
              'features': int(len(features)), 'interpretation': 'SHAP values quantify model-prediction contribution, not causation or injury mechanism.',
              'sample_sha256': hashlib.sha256((OUT / 'shap_sample_index.csv').read_bytes()).hexdigest(),
              'model_sha256': hashlib.sha256((OUT / 'selected_injury_model.joblib').read_bytes()).hexdigest(),
              'top_features': table.head(20).to_dict('records')}
    (OUT / 'shap_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({'rows_explained': len(ids), 'top_features': table.head(10).to_dict('records')}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
