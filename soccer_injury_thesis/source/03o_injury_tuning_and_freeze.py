"""Tune the AP-winning benchmark family/window, validate with LOPO, export.

Selection of family/window from the preceding benchmark is NOT nested here.
The report explicitly treats results as development/internal comparisons, not
an unbiased external estimate. Never overwrites the previous t+1 models.
"""
from pathlib import Path
import argparse
import hashlib
import importlib
import json
import warnings
import contextlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterSampler, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits
from config_soccermon import make_pipeline, BASELINE_PARAMS, TUNING_SPACES, json_safe

bench = importlib.import_module('03n_injury_history_lopo')
OUT = bench.OUT
TUNE = OUT / 'tuning'

def dataset(window):
    d = pd.read_parquet(OUT / f'history_{window}d.parquet')
    features = [c for c in d if c not in ['player_name', 'date', 'injury']]
    return d, features

def group_splits(d):
    # Stratify *athletes* by presence of any positive; never split an athlete.
    subjects = d.groupby('player_name', sort=True).injury.max()
    result = []
    for tr, va in StratifiedKFold(3, shuffle=True, random_state=42).split(subjects.index, subjects):
        a = np.flatnonzero(d.player_name.isin(subjects.index[tr]).to_numpy())
        b = np.flatnonzero(d.player_name.isin(subjects.index[va]).to_numpy())
        assert not set(d.iloc[a].player_name) & set(d.iloc[b].player_name)
        assert d.iloc[a].injury.nunique() == d.iloc[b].injury.nunique() == 2
        result.append((a, b))
    return result

def construct(name, candidate):
    p = make_pipeline(name, candidate['params'])
    estimator = p.named_steps['model']
    params = estimator.get_params()
    if 'n_jobs' in params:
        estimator.set_params(n_jobs=2)
    if 'thread_count' in params:
        estimator.set_params(thread_count=2)
    if name == 'MLP' and 'max_iter' not in candidate['params']:
        estimator.set_params(max_iter=300)
    prep = clone(p[:-1])
    if candidate['strategy'] == 'smote01' and 'scaler' not in prep.named_steps:
        prep = Pipeline(prep.steps + [('scaler', StandardScaler())])
    return prep, estimator

def fit_predict(d, features, train, test, name, candidate, return_bundle=False):
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    prep, estimator = construct(name, candidate)
    y = d.iloc[train].injury.to_numpy()
    with warnings.catch_warnings(record=True) as ws, threadpool_limits(limits=2):
        a = prep.fit_transform(d.iloc[train][features], y)
        fit_kwargs = {}
        strategy = candidate['strategy']
        if strategy == 'ros01':
            a, y = RandomOverSampler(sampling_strategy=.1, random_state=42).fit_resample(a, y)
        elif strategy == 'smote01':
            a, y = SMOTE(sampling_strategy=.1, k_neighbors=min(5, int(y.sum())-1), random_state=42).fit_resample(a, y)
        elif strategy == 'weighted':
            fit_kwargs['sample_weight'] = compute_sample_weight('balanced', y)
        estimator.fit(a, y, **fit_kwargs)
        b = prep.transform(d.iloc[test][features])
        score = estimator.predict_proba(b)[:, 1] if hasattr(estimator, 'predict_proba') else estimator.decision_function(b)
    assert np.isfinite(score).all()
    notes = [str(w.message) for w in ws]
    if return_bundle:
        return score, dict(preprocessor=prep, estimator=estimator, features=features), notes
    return score, notes

def search(d, features, name, candidates):
    splits = group_splits(d)
    results = []
    for candidate in candidates:
        scores, notes = [], []
        for train, test in splits:
            pred, warn = fit_predict(d, features, train, test, name, candidate)
            scores.append(float(average_precision_score(d.iloc[test].injury, pred)))
            notes.extend(warn)
        results.append(dict(candidate_id=candidate['id'], mean_inner_AP=float(np.mean(scores)),
                            inner_fold_AP=scores, warnings=notes))
    best = max(results, key=lambda x: x['mean_inner_AP'])
    return candidates[best['candidate_id']], results

def outer_fold(task):
    fold, player, window, name, candidates = task
    path = TUNE / f'fold_{fold:02d}.csv'
    if path.exists():
        return fold
    with (TUNE / f'fold_{fold:02d}.log').open('w', buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            d, features = dataset(window)
            test = np.flatnonzero(d.player_name.eq(player).to_numpy())
            train = np.flatnonzero(d.player_name.ne(player).to_numpy())
            best, trials = search(d.iloc[train].reset_index(drop=True), features, name, candidates)
            pred, notes = fit_predict(d, features, train, test, name, best)
            result = d.iloc[test][['player_name', 'date', 'injury']].copy()
            result['score'] = pred
            meta = dict(test_player=player, selected=best, inner_search=trials, outer_warnings=notes)
            (TUNE / f'fold_{fold:02d}.json').write_text(json.dumps(json_safe(meta), ensure_ascii=False, indent=2))
            result.to_csv(path.with_suffix('.tmp'), index=False)
            path.with_suffix('.tmp').replace(path)
    return fold

def prepare():
    bench.summarize()
    summary = pd.read_csv(OUT / 'benchmark_summary.csv')
    assert len(summary) == 96, 'Must finish all 8 models x 6 windows x 2 strategies first.'
    summary = summary.sort_values(['AP', 'ROC_AUC'], ascending=False)
    best = summary.iloc[0]
    window_text, name, strategy = best.configuration.split('_')
    window = int(window_text[:-1])
    defaults = dict(BASELINE_PARAMS[name])
    if name == 'MLP':
        defaults['max_iter'] = 300
    candidates = [dict(id=0, strategy=strategy, params=defaults)]
    for s in ['control', 'ros01', 'smote01', 'weighted']:
        if s != strategy:
            candidates.append(dict(id=len(candidates), strategy=s, params=dict(defaults)))
    size = int(np.prod([len(v) for v in TUNING_SPACES[name].values()]))
    random_params = list(ParameterSampler(TUNING_SPACES[name], n_iter=min(12, size), random_state=20261007))
    for i, pars in enumerate(random_params):
        candidates.append(dict(id=len(candidates), strategy=['control', 'ros01', 'smote01', 'weighted'][i % 4],
                               params={k.removeprefix('model__'): v for k, v in pars.items()}))
    spec = dict(window=window, name=name, incumbent=best.configuration, candidates=candidates,
                inner_validation='3 athlete-disjoint folds stratified by athlete positive presence',
                criterion='mean of 3 inner-fold AP values',
                limitation='model family and window preselected from outer benchmark; selection optimism remains')
    TUNE.mkdir(exist_ok=True)
    target = TUNE / 'search_spec.json'
    native = json_safe(spec)
    if target.exists():
        assert json.loads(target.read_text()) == native, 'Search spec changed; do not reuse checkpoints.'
    else:
        target.write_text(json.dumps(native, ensure_ascii=False, indent=2))
    return native, summary

def metrics(frame):
    return dict(AP=float(average_precision_score(frame.injury, frame.score)),
                ROC_AUC=float(roc_auc_score(frame.injury, frame.score)))

def bootstrap(frame, repeats=1000):
    players, codes = np.unique(frame.player_name, return_inverse=True)
    rng = np.random.default_rng(42)
    ap, auc = [], []
    for _ in range(repeats):
        weights = np.bincount(rng.integers(0, len(players), len(players)), minlength=len(players))[codes]
        if frame.loc[weights > 0, 'injury'].nunique() < 2:
            continue
        ap.append(average_precision_score(frame.injury, frame.score, sample_weight=weights))
        auc.append(roc_auc_score(frame.injury, frame.score, sample_weight=weights))
    return dict(valid_replicates=len(ap), AP_95CI=np.quantile(ap, [.025, .975]).tolist(),
                ROC_AUC_95CI=np.quantile(auc, [.025, .975]).tolist(),
                note='athlete-cluster bootstrap of fixed OOF predictions; does not rerun selection/training')

def finalize(spec, summary):
    import platform
    from datetime import datetime, timezone
    from importlib.metadata import version
    files = sorted(TUNE.glob('fold_*.csv'))
    assert len(files) == 50
    tuned = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    assert len(tuned) == 19613 and tuned.injury.sum() == 149 and tuned.player_name.nunique() == 50
    assert not tuned.duplicated(['player_name', 'date']).any()
    original = pd.concat([pd.read_csv(p) for p in sorted((OUT / 'folds' / spec['incumbent']).glob('*.csv'))], ignore_index=True)
    tm, bm = metrics(tuned), metrics(original)
    improved = tm['AP'] > bm['AP']
    d, features = dataset(spec['window'])
    if improved:
        chosen, search_results = search(d, features, spec['name'], spec['candidates'])
    else:
        chosen, search_results = spec['candidates'][0], []
    score, bundle, notes = fit_predict(d, features, np.arange(len(d)), np.arange(min(20, len(d))), spec['name'], chosen, True)
    selected_oof = tuned if improved else original
    details = dict(model=spec['name'], window=spec['window'], selected_candidate=chosen,
                   tuned_selected=improved, benchmark_metrics=bm, tuned_metrics=tm,
                   selected_metrics=metrics(selected_oof), cluster_bootstrap=bootstrap(selected_oof),
                   rows=len(d), positives=int(d.injury.sum()), players=int(d.player_name.nunique()),
                   AP_reference=float(d.injury.mean()), full_training_search=search_results,
                   final_warnings=notes, outcome='original day-level injury notification',
                   features_timing='strictly before outcome date; calendar history',
                   calibration='NOT calibrated; no reliable daily probability claim',
                   selection_limit='Development selection over benchmark configurations and tuning; not unbiased external performance.')
    bundle['metadata'] = json_safe(details)
    joblib.dump(bundle, OUT / 'selected_injury_model.joblib')
    loaded = joblib.load(OUT / 'selected_injury_model.joblib')
    transformed = loaded['preprocessor'].transform(d.iloc[:len(score)][features])
    model = loaded['estimator']
    check = model.predict_proba(transformed)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(transformed)
    np.testing.assert_allclose(score, check)
    selected_oof.to_csv(OUT / 'selected_model_oof_predictions.csv', index=False)
    patient_rows = []
    for player, group in selected_oof.groupby('player_name'):
        patient_rows.append(dict(player_name=player, days=len(group), positives=int(group.injury.sum()),
                                 AP=float(average_precision_score(group.injury, group.score)) if group.injury.sum() else None,
                                 ROC_AUC=float(roc_auc_score(group.injury, group.score)) if group.injury.nunique() == 2 else None))
    pd.DataFrame(patient_rows).to_csv(OUT / 'selected_model_athlete_diagnostics.csv', index=False)
    (OUT / 'selected_model_report.json').write_text(json.dumps(json_safe(details), ensure_ascii=False, indent=2))
    summary['model'] = summary.configuration.map(lambda x: x.split('_')[1])
    winners = summary.sort_values('AP', ascending=False).groupby('model', sort=False).head(1)
    winners.to_csv(OUT / 'eight_model_best_configurations.csv', index=False)
    report = ['# 原始injury模型重跑结果', '',
              f'共同样本：50名运动员，{len(d)}个运动员日，149阳性。结局为原始日级伤病通知，不是临床确诊新发损伤。', '',
              '## 八模型各自最佳配置（开发集LOPO比较）', '',
              '|配置|AP|ROC-AUC|', '|---|---:|---:|']
    for row in winners.itertuples():
        report.append(f'|{row.configuration}|{row.AP:.6f}|{row.ROC_AUC:.6f}|')
    report += ['', f'选定模型：{spec["name"]}，{spec["window"]}自然日历史；是否采用调参结果：{improved}。',
               f'初始优胜配置AP={bm["AP"]:.6f}；该家族/窗口训练折内部调参后AP={tm["AP"]:.6f}。',
               '调参并不保证提升；只有调参后的LOPO AP更高才替换原配置。', '',
               '所有插补、缩放、重采样、权重计算及内层调参均排除外层测试运动员。',
               '模型家族/窗口先经全体LOPO比较筛选，因此最终最高分仍存在选择乐观性；不能称为无偏外部验证或所有可能模型的最优解。',
               '导出的模型是未校准的源域候选模型，原始输出不能直接解释成可信的临床损伤概率。', '',
               '模型文件：selected_injury_model.joblib；逐日外层预测：selected_model_oof_predictions.csv。',
               '完整搜索、逐折选择和警告见tuning目录；源数据与纳入审计见data_audit.json。', '',
               '方法依据：https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html',
               '重采样边界：https://imbalanced-learn.org/stable/common_pitfalls.html']
    (OUT / 'FINAL_REPORT.md').write_text('\n'.join(report))
    sources = [bench.SOURCE, bench.ROOT / 'config_soccermon.py',
               bench.ROOT / '03n_injury_history_lopo.py', Path(__file__).resolve()]
    sources += [OUT / f'history_{w}d.parquet' for w in bench.WINDOWS]
    manifest = dict(completed_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
                    versions={name: version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'imbalanced-learn', 'xgboost', 'lightgbm', 'catboost', 'joblib']},
                    input_and_code_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    outcome='original injury_report_t copied to injury; never shifted',
                    benchmark_complete_configurations=96, benchmark_complete_outer_fits=4800,
                    tuning_candidates=len(spec['candidates']), tuning_complete_outer_folds=50,
                    model_sha256=hashlib.sha256((OUT/'selected_injury_model.joblib').read_bytes()).hexdigest())
    (OUT / 'RUN_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(json_safe({k:v for k,v in details.items() if k not in ['full_training_search','final_warnings']}), ensure_ascii=False, indent=2), flush=True)

def main():
    import fcntl
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    with (OUT / 'tuning.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec, summary = prepare()
        d, _ = dataset(spec['window'])
        tasks = [(i, p, spec['window'], spec['name'], spec['candidates']) for i, p in enumerate(sorted(d.player_name.unique()))]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(outer_fold, task) for task in tasks]
            for future in as_completed(futures):
                print(f'[tuning] outer fold {future.result()+1}/50 saved', flush=True)
        finalize(spec, summary)

if __name__ == '__main__':
    main()
