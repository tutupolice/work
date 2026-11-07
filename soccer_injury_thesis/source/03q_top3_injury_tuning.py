"""Tune the top-three AP configurations from the injury benchmark.

Each model/window/strategy is tuned separately. Hyperparameter selection is
inside each outer training set using athlete-disjoint inner folds; the outer
test athlete is never used for fitting or selection.
"""
from pathlib import Path
import contextlib
import hashlib
import importlib
import json
import fcntl
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

base = importlib.import_module('03o_injury_tuning_and_freeze')
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/injury_history_lopo_v1'
TOP = OUT / 'top3_tuning'
BENCH = pd.read_csv(OUT / 'benchmark_summary.csv')

def parse_config(config):
    window, model, strategy = config.split('_', 2)
    return int(window[:-1]), model, strategy

def make_candidates(model, strategy):
    defaults = dict(base.BASELINE_PARAMS[model])
    if model == 'MLP':
        defaults['max_iter'] = 300
    candidates = []
    for s in ['control', 'ros01', 'smote01', 'weighted']:
        candidates.append(dict(id=len(candidates), strategy=s, params=dict(defaults)))
    space = base.TUNING_SPACES[model]
    size = int(np.prod([len(v) for v in space.values()]))
    sampled = list(ParameterSampler(space, n_iter=min(12, size), random_state=20261007))
    for pars in sampled:
        candidates.append(dict(id=len(candidates), strategy=['control', 'ros01', 'smote01', 'weighted'][len(candidates) % 4],
                               params={k.removeprefix('model__'): v for k, v in pars.items()}))
    return candidates

def config_dir(config):
    return TOP / config.replace('-', '_')

def run_outer(task):
    config, fold, player, candidates = task
    window, model, _ = parse_config(config)
    folder = config_dir(config)
    folder.mkdir(parents=True, exist_ok=True)
    pred_path = folder / f'fold_{fold:02d}.csv'
    if pred_path.exists():
        return config, fold
    log_path = folder / f'fold_{fold:02d}.log'
    with log_path.open('w', buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        d, features = base.dataset(window)
        test = np.flatnonzero(d.player_name.eq(player).to_numpy())
        train = np.flatnonzero(d.player_name.ne(player).to_numpy())
        train_d = d.iloc[train].reset_index(drop=True)
        best, trials = base.search(train_d, features, model, candidates)
        pred, warnings = base.fit_predict(d, features, train, test, model, best)
        frame = d.iloc[test][['player_name', 'date', 'injury']].copy()
        frame['score'] = pred
        frame.to_csv(pred_path.with_suffix('.tmp'), index=False)
        pred_path.with_suffix('.tmp').replace(pred_path)
        meta = {'config': config, 'fold': fold, 'test_player': player, 'selected': best,
                'inner_search': trials, 'outer_warnings': warnings}
        (folder / f'fold_{fold:02d}.json').write_text(json.dumps(base.json_safe(meta), ensure_ascii=False, indent=2))
    return config, fold

def summarize_config(config):
    folder = config_dir(config)
    files = sorted(folder.glob('fold_*.csv'))
    if len(files) != 50:
        return None
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    assert len(d) == 19613 and int(d.injury.sum()) == 149 and d.player_name.nunique() == 50
    return dict(configuration=config, rows=len(d), positives=int(d.injury.sum()),
                tuned_AP=float(average_precision_score(d.injury, d.score)),
                tuned_ROC_AUC=float(roc_auc_score(d.injury, d.score)),
                tuned_Brier=float(brier_score_loss(d.injury, d.score)),
                mean_score=float(d.score.mean()))

def main():
    lock_path = TOP / 'top3_tuning.lock'
    TOP.mkdir(exist_ok=True)
    with lock_path.open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        top = BENCH.sort_values('AP', ascending=False).head(3).copy()
        specs = []
        for config in top.configuration:
            window, model, strategy = parse_config(config)
            specs.append({'configuration': config, 'window': window, 'model': model,
                          'initial_strategy': strategy, 'initial_AP': float(top.loc[top.configuration.eq(config), 'AP'].iloc[0]),
                          'initial_ROC_AUC': float(top.loc[top.configuration.eq(config), 'ROC_AUC'].iloc[0]),
                          'candidates': make_candidates(model, strategy)})
        (TOP / 'top3_search_spec.json').write_text(json.dumps(base.json_safe(specs), ensure_ascii=False, indent=2))
        tasks = []
        for spec in specs:
            d, _ = base.dataset(spec['window'])
            players = sorted(d.player_name.unique())
            tasks.extend((spec['configuration'], i, p, spec['candidates']) for i, p in enumerate(players))
        with ProcessPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(run_outer, task) for task in tasks]
            done = 0
            for future in as_completed(futures):
                config, fold = future.result()
                done += 1
                if done % 10 == 0 or fold == 49:
                    print(f'[top3 tuning] {done}/150 folds; latest {config} {fold+1}/50', flush=True)
        rows = []
        for spec in specs:
            result = summarize_config(spec['configuration'])
            assert result is not None
            result.update({k: spec[k] for k in ['window', 'model', 'initial_strategy', 'initial_AP', 'initial_ROC_AUC']})
            result['AP_delta'] = result['tuned_AP'] - result['initial_AP']
            result['ROC_AUC_delta'] = result['tuned_ROC_AUC'] - result['initial_ROC_AUC']
            rows.append(result)
        out = pd.DataFrame(rows).sort_values('tuned_AP', ascending=False)
        out.to_csv(TOP / 'top3_tuning_summary.csv', index=False)
        report = ['# injury标签AP前三模型调参结果', '',
                  '每个模型均在其自身外层训练运动员内部进行3折运动员分组内层搜索，外层测试运动员不参与插补、标准化、重采样或参数选择。', '',
                  '|配置|初始AP|调参后AP|AP变化|初始ROC-AUC|调参后ROC-AUC|', '|---|---:|---:|---:|---:|---:|']
        for row in out.itertuples():
            report.append(f'|{row.configuration}|{row.initial_AP:.6f}|{row.tuned_AP:.6f}|{row.AP_delta:+.6f}|{row.initial_ROC_AUC:.6f}|{row.tuned_ROC_AUC:.6f}|')
        winner = out.iloc[0]
        report += ['', f'按调参后AP，当前前三候选中的最高配置为{winner.configuration}，AP={winner.tuned_AP:.6f}。',
                   '该结果仍是开发阶段模型选择结果；在同一来源数据上选择模型、窗口和训练策略会带来选择乐观性，后续外部验证仍然必要。',
                   '逐折预测、每折内层搜索结果和警告保存在对应配置子目录。']
        (TOP / 'TOP3_TUNING_REPORT.md').write_text('\n'.join(report))
        manifest = {'benchmark_sha256': hashlib.sha256((OUT/'benchmark_summary.csv').read_bytes()).hexdigest(),
                    'top3': [s['configuration'] for s in specs], 'outer_folds': 150,
                    'rows_per_configuration': 19613, 'positives_per_configuration': 149}
        (TOP / 'top3_tuning_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        print(out.to_string(index=False), flush=True)

if __name__ == '__main__':
    main()
