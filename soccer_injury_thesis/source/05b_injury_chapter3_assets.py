"""Generate reproducible Chapter 3 figures from the injury restart outputs."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.calibration import calibration_curve

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/injury_history_lopo_v1'
FIG = OUT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)

def save(fig, name):
    fig.savefig(FIG / f'{name}.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG / f'{name}.svg', bbox_inches='tight')
    plt.close(fig)

def main():
    bench = pd.read_csv(OUT / 'benchmark_summary.csv')
    best = pd.read_csv(OUT / 'eight_model_best_configurations.csv')
    cal = pd.read_csv(OUT / 'selected_model_calibrated_oof_predictions.csv')
    y = cal.injury.to_numpy()
    random_ap = float(y.mean())

    rank = best.sort_values('AP', ascending=True)
    labels = rank.configuration.str.replace('_', ' ', regex=False)
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.barh(labels, rank.AP, color='#6F8FB2')
    ax.axvline(random_ap, color='#B23A48', linestyle='--', linewidth=1.3, label=f'Random AP = {random_ap:.4f}')
    ax.set_xlabel('Average precision (AP)')
    ax.set_ylabel('Best configuration for each model family')
    ax.set_title('Best AP configuration within each model family')
    ax.legend(frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    save(fig, 'fig3_4_model_window_ap_comparison')

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    for label, score, color in [('Raw LightGBM output', cal.raw_probability, '#6F8FB2'),
                                ('Nested calibrated output', cal.calibrated_probability, '#245B8F')]:
        p, r, _ = precision_recall_curve(y, score)
        ax.plot(r, p, label=label, color=color, linewidth=1.8)
    ax.axhline(random_ap, color='#B23A48', linestyle='--', linewidth=1.2, label=f'Random AP baseline = {random_ap:.4f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision–recall curves for the selected source-domain model')
    ax.legend(frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    save(fig, 'fig3_5_precision_recall_selected_model')

    fig, ax = plt.subplots(figsize=(6.3, 5.4))
    calibration_points = []
    for label, score, color in [('Raw LightGBM output', cal.raw_probability, '#6F8FB2'),
                                ('Nested calibrated output', cal.calibrated_probability, '#245B8F')]:
        frac, mean = calibration_curve(y, score, n_bins=10, strategy='quantile')
        calibration_points.extend(mean.tolist())
        calibration_points.extend(frac.tolist())
        ax.plot(mean, frac, marker='o', label=label, color=color)
    upper = min(1.0, max(calibration_points) * 1.15)
    ax.plot([0, upper], [0, upper], linestyle='--', color='#777777', linewidth=1.0, label='Perfect calibration')
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Observed event frequency')
    ax.set_title('Nested probability calibration')
    ax.legend(frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    save(fig, 'fig3_7_probability_calibration')

    summary = {
        'source': 'outputs/injury_history_lopo_v1/benchmark_summary.csv',
        'calibration_source': 'outputs/injury_history_lopo_v1/selected_model_calibrated_oof_predictions.csv',
        'plots': ['fig3_4_model_window_ap_comparison', 'fig3_5_precision_recall_selected_model', 'fig3_7_probability_calibration'],
        'random_ap': random_ap,
        'best_configuration_by_AP': bench.sort_values('AP', ascending=False).iloc[0].to_dict(),
        'calibration_metrics': json.loads((OUT / 'calibration_report.json').read_text()),
        'figure_input_sha256': {
            'benchmark_summary.csv': hashlib.sha256((OUT / 'benchmark_summary.csv').read_bytes()).hexdigest(),
            'selected_model_calibrated_oof_predictions.csv': hashlib.sha256((OUT / 'selected_model_calibrated_oof_predictions.csv').read_bytes()).hexdigest(),
            'shap_global_feature_contributions.csv': hashlib.sha256((OUT / 'shap_global_feature_contributions.csv').read_bytes()).hexdigest(),
        },
        'figure_boundary': 'All numerical curves and marks are generated from actual LOPO predictions; no simulated points or manually edited values.',
    }
    (OUT / 'chapter3_assets_manifest.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
