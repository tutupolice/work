"""EDA for the original day-level injury label restart."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/injury_history_lopo_v1'
FIG = OUT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)

def save(fig, name):
    fig.savefig(FIG / f'{name}.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG / f'{name}.svg', bbox_inches='tight')
    plt.close(fig)

def main():
    d = pd.read_parquet(OUT / 'history_8d.parquet')
    d['date'] = pd.to_datetime(d['date'])
    source = pd.read_parquet(OUT / 'merged_source_snapshot.parquet')
    source['date'] = pd.to_datetime(source['date'])
    source['year_month'] = source.date.dt.to_period('M').astype(str)
    d['year_month'] = d.date.dt.to_period('M').astype(str)

    athlete = d.groupby('player_name', as_index=False).agg(
        days=('injury', 'size'), positive_days=('injury', 'sum'),
        positive_rate=('injury', 'mean'))
    athlete['has_positive'] = athlete.positive_days.gt(0)
    monthly = d.groupby('year_month', as_index=False).agg(
        days=('injury', 'size'), positive_days=('injury', 'sum'),
        positive_rate=('injury', 'mean'))
    missing = pd.DataFrame({
        'field': ['source_panel', 'analysis_8d_history'],
        'rows': [len(source), len(d)],
        'players': [source.player_name.nunique(), d.player_name.nunique()],
        'positive_days': [int(source.injury_report_t.sum()), int(d.injury.sum())],
        'positive_rate': [float(source.injury_report_t.mean()), float(d.injury.mean())],
    })
    athlete.to_csv(OUT / 'eda_athlete_distribution.csv', index=False)
    monthly.to_csv(OUT / 'eda_monthly_distribution.csv', index=False)
    missing.to_csv(OUT / 'eda_sample_summary.csv', index=False)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(['Negative days', 'Positive injury days'], [int((d.injury == 0).sum()), int(d.injury.sum())], color=['#B7C9E2', '#245B8F'])
    ax.set_ylabel('Athlete-days')
    ax.set_title('Distribution of the original day-level injury label')
    ax.spines[['top', 'right']].set_visible(False)
    save(fig, 'fig3_1_outcome_distribution')

    ordered = athlete.sort_values('positive_days', ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(np.arange(len(ordered)), ordered.positive_days, color=np.where(ordered.has_positive, '#245B8F', '#B7C9E2'))
    ax.set_xlabel('Athlete rank (sorted by positive days)')
    ax.set_ylabel('Positive injury days')
    ax.set_title('Positive injury days by athlete')
    ax.set_xticks([])
    ax.spines[['top', 'right']].set_visible(False)
    save(fig, 'fig3_2_positive_days_by_athlete')

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(pd.to_datetime(monthly.year_month), monthly.positive_days, marker='o', color='#245B8F')
    ax.set_xlabel('Calendar month')
    ax.set_ylabel('Positive injury days')
    ax.set_title('Temporal distribution of positive injury days')
    ax.tick_params(axis='x', rotation=45)
    ax.spines[['top', 'right']].set_visible(False)
    save(fig, 'fig3_3_monthly_positive_days')

    audit = {
        'source_snapshot': str(OUT / 'merged_source_snapshot.parquet'),
        'analysis_file': str(OUT / 'history_8d.parquet'),
        'analysis': 'EDA uses original injury_report_t as same-day injury label; no t+1 shift',
        'source_sha256': hashlib.sha256((OUT / 'merged_source_snapshot.parquet').read_bytes()).hexdigest(),
        'analysis_sha256': hashlib.sha256((OUT / 'history_8d.parquet').read_bytes()).hexdigest(),
        'rows': int(len(d)), 'players': int(d.player_name.nunique()),
        'positive_days': int(d.injury.sum()), 'positive_rate': float(d.injury.mean()),
        'positive_players': int(athlete.has_positive.sum()),
        'figures': ['fig3_1_outcome_distribution', 'fig3_2_positive_days_by_athlete', 'fig3_3_monthly_positive_days'],
    }
    (OUT / 'eda_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps(audit, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
