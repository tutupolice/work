"""Independent restart: original injury on date d, predictors strictly before d.

Run with pmsys Python. Never modifies previous data/results. Each outer fold is
checkpointed. No test-player data fit the imputer, scaler, or oversampler.
"""
from pathlib import Path
import argparse
import hashlib
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.base import clone
from threadpoolctl import threadpool_limits
from config_soccermon import make_pipeline, MODEL_NAMES, SOURCE_COMMON_CURRENT

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/injury_history_lopo_v1'
SOURCE = ROOT / 'verified_outputs/01_source_preprocessing_v1/source_athlete_day_preprocessed.parquet'
WINDOWS = (2, 4, 8, 16, 21, 32)
FIELDS = SOURCE_COMMON_CURRENT + ['mood_raw', 'readiness_raw', 'daily_load_au']
FLAGS = ['has_training_record', 'has_wellness_record', 'has_gps_record']

def build_features():
    OUT.mkdir(parents=True, exist_ok=True)
    p = pd.read_parquet(SOURCE).sort_values(['player_name', 'date'])
    assert p.player_name.nunique() == 50
    assert not p.duplicated(['player_name', 'date']).any()
    assert p.injury_report_t.isin([0, 1]).all()
    lookup = pd.read_csv(ROOT / 'verified_outputs/07_source_injury_reportday_t1_v1/04_injury_report_day.csv')
    lookup['date'] = pd.to_datetime(lookup.date)
    check = p[['player_name', 'date', 'injury_report_t']].merge(
        lookup[['player_name', 'date', 'injury_report']], how='left',
        on=['player_name', 'date'], validate='one_to_one')
    assert np.array_equal(check.injury_report_t, check.injury_report.fillna(0))
    parts = {w: [] for w in WINDOWS}
    for player, g in p.groupby('player_name', sort=True):
        calendar = pd.date_range(g.date.min(), g.date.max(), freq='D')
        daily = g.set_index('date').reindex(calendar)
        values = daily[FIELDS].apply(pd.to_numeric).replace([np.inf, -np.inf], np.nan)
        # Shift by one CALENDAR day; no feature sees the target day.
        past = values.shift(1)
        observed = values.notna().any(axis=1).astype(float).shift(1).fillna(0)
        for w in WINDOWS:
            frame = pd.DataFrame(index=calendar)
            for field in FIELDS:
                r = past[field].rolling(w, min_periods=1)
                frame[field + '__mean'] = r.mean()
                frame[field + '__std'] = r.std(ddof=0)
                frame[field + '__max'] = r.max()
                frame[field + '__coverage'] = r.count() / w
                # Last calendar-day observation; missing is not forward-filled.
                frame[field + '__previous_day'] = past[field]
            for flag in FLAGS:
                frame[flag + '__rate'] = daily[flag].fillna(0).shift(1).rolling(w, min_periods=1).sum() / w
            frame['history_observed_days'] = observed.rolling(w, min_periods=1).sum()
            frame['history_calendar_coverage'] = np.minimum(np.arange(len(calendar)), w) / w
            frame = frame.reindex(pd.DatetimeIndex(g.date)).copy()
            frame['player_name'] = player
            frame['date'] = frame.index
            frame['injury'] = g.injury_report_t.to_numpy()
            frame['eligible'] = frame.history_observed_days.ge(1)
            parts[w].append(frame.reset_index(drop=True))
    matrices = {w: pd.concat(v, ignore_index=True) for w, v in parts.items()}
    # Same label-independent cohort; shorter windows may contain missing data.
    common = matrices[32].eligible.to_numpy()
    excluded = matrices[2].loc[~common, ['player_name', 'date', 'injury']]
    excluded.to_csv(OUT / 'excluded_no_32day_history.csv', index=False)
    for w, frame in matrices.items():
        assert np.array_equal(frame.injury, matrices[2].injury)
        frame.loc[common].drop(columns='eligible').to_parquet(OUT / f'history_{w}d.parquet', index=False)
    p.to_parquet(OUT / 'merged_source_snapshot.parquet', index=False)
    included = matrices[2].loc[common]
    audit = dict(source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                 target='original injury_report_t renamed injury; no label shift',
                 input_interval='[date-window, date-1] calendar days',
                 inclusion='existing merged panel row, at least one observed input in preceding 32 calendar days; common cohort for all windows',
                 missing_label_meaning='no submitted notification in source report-day lookup, not confirmed clinical health',
                 source_rows=len(p), source_positives=int(p.injury_report_t.sum()),
                 included_rows=len(included), included_positives=int(included.injury.sum()),
                 players=int(included.player_name.nunique()),
                 excluded_rows=len(excluded), excluded_positives=int(excluded.injury.sum()),
                 fields=FIELDS, windows=list(WINDOWS),
                 imbalance='unresampled control and training-only random oversampling to minority/majority=0.1; not probability calibrated')
    (OUT / 'data_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps(audit, ensure_ascii=False), flush=True)

def benchmark(selected_models, windows, strategies, update_summary=True):
    from imblearn.over_sampling import RandomOverSampler
    for w in windows:
        d = pd.read_parquet(OUT / f'history_{w}d.parquet')
        features = [c for c in d if c not in ['date', 'player_name', 'injury']]
        X, y = d[features], d.injury.to_numpy()
        for name in selected_models:
            for strategy in strategies:
                folder = OUT / 'folds' / f'{w}d_{name}_{strategy}'
                folder.mkdir(parents=True, exist_ok=True)
                for fold, player in enumerate(sorted(d.player_name.unique())):
                    dest = folder / f'{fold:02d}.csv'
                    if dest.exists():
                        continue
                    test = d.player_name.eq(player).to_numpy()
                    train = ~test
                    pipeline = make_pipeline(name)
                    model = pipeline.named_steps['model']
                    mp = model.get_params()
                    if 'n_jobs' in mp:
                        model.set_params(n_jobs=2)
                    if 'thread_count' in mp:
                        model.set_params(thread_count=2)
                    if name == 'MLP':
                        model.set_params(max_iter=300)
                    preprocess = clone(pipeline[:-1])
                    with warnings.catch_warnings(record=True) as ws, threadpool_limits(limits=2):
                        a = preprocess.fit_transform(X.loc[train], y[train])
                        b = preprocess.transform(X.loc[test])
                        labels = y[train]
                        if strategy == 'ros01':
                            a, labels = RandomOverSampler(sampling_strategy=0.1, random_state=42).fit_resample(a, labels)
                        model.fit(a, labels)
                        score = model.predict_proba(b)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(b)
                    result = d.loc[test, ['player_name', 'date', 'injury']].copy()
                    result['score'] = score
                    result.to_csv(dest.with_suffix('.tmp'), index=False)
                    dest.with_suffix('.tmp').replace(dest)
                    (folder / f'{fold:02d}_warnings.json').write_text(json.dumps([str(v.message) for v in ws], ensure_ascii=False))
                    print(f'[injury] {w}d {name} {strategy} LOPO {fold+1}/50 complete', flush=True)
                if update_summary:
                    summarize()

def run_configuration(task):
    import contextlib
    w, name, strategy = task
    logdir = OUT / 'logs'
    logdir.mkdir(exist_ok=True)
    with (logdir / f'{w}d_{name}_{strategy}.log').open('a', buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            benchmark([name], [w], [strategy], update_summary=False)
    return f'{w}d_{name}_{strategy}'

def parallel_benchmark(models, windows, strategies, workers):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import fcntl
    import os
    # Only this coordinator writes the summary; no overlapping launches.
    with (OUT / 'benchmark.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.write(str(os.getpid()))
        lock.flush()
        tasks = [(w, m, s) for w in windows for m in models for s in strategies]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_configuration, t): t for t in tasks}
            for future in as_completed(futures):
                completed = future.result()
                summarize()
                print(f'[COMPLETE] {completed} (50/50 LOPO)', flush=True)

def summarize():
    rows = []
    for folder in sorted((OUT / 'folds').glob('*')):
        files = sorted(folder.glob('*.csv'))
        if len(files) != 50:
            continue
        d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        assert d.player_name.nunique() == 50
        assert not d.duplicated(['player_name', 'date']).any()
        assert np.isfinite(d.score).all()
        assert len(d) == 19613 and int(d.injury.sum()) == 149
        rows.append(dict(configuration=folder.name, rows=len(d), positives=int(d.injury.sum()),
                         AP=average_precision_score(d.injury, d.score),
                         ROC_AUC=roc_auc_score(d.injury, d.score)))
    if rows:
        pd.DataFrame(rows).sort_values('AP', ascending=False).to_csv(OUT / 'benchmark_summary.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['features', 'benchmark', 'all'], default='all')
    parser.add_argument('--models', nargs='+', default=list(MODEL_NAMES), choices=list(MODEL_NAMES))
    parser.add_argument('--windows', type=int, nargs='+', default=list(WINDOWS), choices=list(WINDOWS))
    parser.add_argument('--strategies', nargs='+', default=['control', 'ros01'], choices=['control', 'ros01'])
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.stage in ['features', 'all']:
        build_features()
    if args.stage in ['benchmark', 'all']:
        parallel_benchmark(args.models, args.windows, args.strategies, args.workers)
