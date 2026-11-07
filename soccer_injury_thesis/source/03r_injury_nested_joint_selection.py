"""Nested joint selection for the original day-level injury-notification task.

This program estimates the performance of the *selection procedure*, rather
than reusing the previous LOPO results to declare a winner.  In each outer
LOPO fold, one athlete is held out completely.  Model family and history
window are selected only inside the remaining athletes by three
athlete-disjoint inner folds, while natural prevalence is fixed as the
pre-specified primary training strategy.  The selected configuration is then fit
on all outer-training athletes and evaluated once on the held-out athlete.

The program intentionally uses the fixed benchmark parameter sets from
03n.  It therefore estimates the nested performance of the pre-specified
model/window/strategy comparison.  Hyperparameter tuning is a separate
development step and must not be described as part of this unbiased estimate
unless it is nested in the same outer loop.

The previous outputs are never overwritten.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

import importlib

bench = importlib.import_module("03n_injury_history_lopo")
tune = importlib.import_module("03o_injury_tuning_and_freeze")
from config_soccermon import BASELINE_PARAMS, MODEL_NAMES, json_safe


ROOT = Path(__file__).resolve().parent
SOURCE_OUT = ROOT / "outputs" / "injury_history_lopo_v1"
OUT = SOURCE_OUT / "nested_joint_selection"
WINDOWS = (2, 4, 8, 16, 21, 32)
# Natural prevalence is the pre-specified primary analysis.  The existing
# 03n ROS results remain a sensitivity analysis and are not allowed to compete
# for the primary champion in the nested estimate.
STRATEGIES = ("control",)
RANDOM_STATE = 20261007


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in {"player_name", "date", "injury"}]


def make_candidates() -> list[dict]:
    candidates = []
    for window in WINDOWS:
        for model in MODEL_NAMES:
            params = dict(BASELINE_PARAMS[model])
            if model == "MLP":
                params["max_iter"] = 300
            for strategy in STRATEGIES:
                candidates.append(
                    {
                        "id": len(candidates),
                        "window": window,
                        "model": model,
                        "strategy": strategy,
                        "params": params,
                    }
                )
    assert len(candidates) == 48
    return candidates


def athlete_inner_splits(frame: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create three athlete-disjoint inner folds stratified by athlete outcome presence."""
    subject = frame.groupby("player_name", sort=True).injury.max()
    if subject.nunique() != 2 or subject.sum() < 3 or (len(subject) - subject.sum()) < 3:
        raise RuntimeError("Inner athlete split cannot preserve both outcome classes.")
    skf = StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE)
    splits = []
    for tr_subject, va_subject in skf.split(subject.index, subject.to_numpy()):
        train_players = set(subject.index[tr_subject])
        valid_players = set(subject.index[va_subject])
        train = np.flatnonzero(frame.player_name.isin(train_players).to_numpy())
        valid = np.flatnonzero(frame.player_name.isin(valid_players).to_numpy())
        assert not train_players & valid_players
        assert frame.iloc[train].injury.nunique() == 2
        assert frame.iloc[valid].injury.nunique() == 2
        splits.append((train, valid))
    return splits


def evaluate_candidate(candidate: dict, frames: dict[int, pd.DataFrame], outer_train_players: set[str]) -> dict:
    """Evaluate one fixed candidate by athlete-disjoint inner AP/ROC-AUC."""
    frame = frames[candidate["window"]]
    train_frame = frame[frame.player_name.isin(outer_train_players)].reset_index(drop=True)
    features = feature_columns(train_frame)
    splits = athlete_inner_splits(train_frame)
    aps, aucs, warning_messages = [], [], []
    for train, valid in splits:
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=2):
            score, notes = tune.fit_predict(
                train_frame,
                features,
                train,
                valid,
                candidate["model"],
                {"strategy": candidate["strategy"], "params": candidate["params"]},
            )
        y_valid = train_frame.iloc[valid].injury.to_numpy()
        aps.append(float(average_precision_score(y_valid, score)))
        aucs.append(float(roc_auc_score(y_valid, score)))
        warning_messages.extend(notes)
        warning_messages.extend(str(w.message) for w in caught)
    return {
        "candidate_id": candidate["id"],
        "window": candidate["window"],
        "model": candidate["model"],
        "strategy": candidate["strategy"],
        "inner_AP_mean": float(np.mean(aps)),
        "inner_ROC_AUC_mean": float(np.mean(aucs)),
        "inner_AP_folds": aps,
        "inner_ROC_AUC_folds": aucs,
        "warnings": sorted(set(warning_messages)),
    }


def choose(results: list[dict]) -> dict:
    """AP is primary; ROC-AUC is only a deterministic tie-breaker."""
    return max(results, key=lambda r: (r["inner_AP_mean"], r["inner_ROC_AUC_mean"]))


def run_outer_fold(fold: int, test_player: str, candidates: list[dict]) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fold_csv = OUT / f"outer_fold_{fold:02d}_predictions.csv"
    fold_json = OUT / f"outer_fold_{fold:02d}_selection.json"
    if fold_csv.exists() and fold_json.exists():
        return fold

    frames = {w: pd.read_parquet(SOURCE_OUT / f"history_{w}d.parquet") for w in WINDOWS}
    reference = frames[32]
    players = sorted(reference.player_name.unique())
    assert test_player in players
    outer_train_players = set(players) - {test_player}
    results = [evaluate_candidate(c, frames, outer_train_players) for c in candidates]
    selected_result = choose(results)
    selected = candidates[selected_result["candidate_id"]]

    frame = frames[selected["window"]]
    features = feature_columns(frame)
    train = np.flatnonzero(frame.player_name.ne(test_player).to_numpy())
    test = np.flatnonzero(frame.player_name.eq(test_player).to_numpy())
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=2):
        score, notes = tune.fit_predict(
            frame,
            features,
            train,
            test,
            selected["model"],
            {"strategy": selected["strategy"], "params": selected["params"]},
        )
    prediction = frame.iloc[test][["player_name", "date", "injury"]].copy()
    prediction["score"] = score
    prediction["selected_configuration"] = (
        f"{selected['window']}d_{selected['model']}_{selected['strategy']}"
    )
    prediction.to_csv(fold_csv.with_suffix(".tmp"), index=False)
    fold_csv.with_suffix(".tmp").replace(fold_csv)

    meta = {
        "fold": fold,
        "test_player": test_player,
        "selected_configuration": selected,
        "selected_inner_result": selected_result,
        "inner_candidate_results": results,
        "outer_warnings": notes + [str(w.message) for w in caught],
        "selection_rule": "max mean inner AP; mean inner ROC-AUC tie-breaker",
    }
    fold_json.write_text(json.dumps(json_safe(meta), ensure_ascii=False, indent=2))
    return fold


def cluster_bootstrap(frame: pd.DataFrame, repeats: int = 2000) -> dict:
    players, codes = np.unique(frame.player_name, return_inverse=True)
    rng = np.random.default_rng(42)
    ap, auc = [], []
    for _ in range(repeats):
        sampled = rng.integers(0, len(players), len(players))
        weights = np.bincount(sampled, minlength=len(players))[codes]
        valid = weights > 0
        if frame.loc[valid, "injury"].nunique() < 2:
            continue
        ap.append(average_precision_score(frame.injury, frame.score, sample_weight=weights))
        auc.append(roc_auc_score(frame.injury, frame.score, sample_weight=weights))
    return {
        "valid_replicates": len(ap),
        "AP_95CI": np.quantile(ap, [0.025, 0.975]).tolist(),
        "ROC_AUC_95CI": np.quantile(auc, [0.025, 0.975]).tolist(),
        "note": "athlete-cluster bootstrap of nested outer predictions; selection is not rerun",
    }


def summarize(candidates: list[dict]) -> None:
    files = sorted(OUT.glob("outer_fold_*_predictions.csv"))
    metas = sorted(OUT.glob("outer_fold_*_selection.json"))
    assert len(files) == 50 and len(metas) == 50
    pred = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    assert pred.player_name.nunique() == 50
    assert not pred.duplicated(["player_name", "date"]).any()
    assert len(pred) == 19613 and int(pred.injury.sum()) == 149
    pred.to_csv(OUT / "nested_joint_oof_predictions.csv", index=False)

    selected_rows = []
    for path in metas:
        item = json.loads(path.read_text())
        s = item["selected_configuration"]
        selected_rows.append(
            {
                "fold": item["fold"],
                "test_player": item["test_player"],
                "configuration": f"{s['window']}d_{s['model']}_{s['strategy']}",
                "window": s["window"],
                "model": s["model"],
                "strategy": s["strategy"],
                "inner_AP_mean": item["selected_inner_result"]["inner_AP_mean"],
                "inner_ROC_AUC_mean": item["selected_inner_result"]["inner_ROC_AUC_mean"],
            }
        )
    selected = pd.DataFrame(selected_rows).sort_values("fold")
    selected.to_csv(OUT / "outer_fold_selected_configurations.csv", index=False)

    counts = selected.configuration.value_counts().rename_axis("configuration").reset_index(name="outer_selection_count")
    metrics = {
        "AP": float(average_precision_score(pred.injury, pred.score)),
        "ROC_AUC": float(roc_auc_score(pred.injury, pred.score)),
        "AP_reference": float(pred.injury.mean()),
        "rows": int(len(pred)),
        "positives": int(pred.injury.sum()),
        "players": int(pred.player_name.nunique()),
        "cluster_bootstrap": cluster_bootstrap(pred),
        "selection_counts": counts.to_dict(orient="records"),
    }
    (OUT / "nested_joint_report.json").write_text(json.dumps(json_safe(metrics), ensure_ascii=False, indent=2))

    report = [
        "# 原始injury标签联合嵌套LOPO结果",
        "",
        "本分析在每个外层LOPO训练集中选择模型和历史窗口，类别自然比例作为预先规定的主分析策略，再对外层留出运动员进行一次预测。",
        "外层测试运动员不参与插补、缩放、重采样、配置选择或参数拟合。",
        "",
        f"分析样本：{len(pred)}条运动员日，{int(pred.injury.sum())}个阳性，{pred.player_name.nunique()}名运动员；AP随机基线={pred.injury.mean():.6f}。",
        "",
        "## 嵌套外层性能",
        "",
        f"AP={metrics['AP']:.6f}；ROC-AUC={metrics['ROC_AUC']:.6f}。",
        f"按运动员聚类bootstrap的AP 95%CI={metrics['cluster_bootstrap']['AP_95CI']}；ROC-AUC 95%CI={metrics['cluster_bootstrap']['ROC_AUC_95CI']}。",
        "",
        "## 外层选择频数",
        "",
        "|配置|外层被选择次数|",
        "|---|---:|",
    ]
    for row in counts.itertuples(index=False):
        report.append(f"|{row.configuration}|{int(row.outer_selection_count)}|")
    report += [
        "",
        "本结果估计的是预先固定的48配置选择程序，而不是某一个配置在同一数据集上被挑选后的表观性能。",
        "固定模型参数采用03n基准参数；若将超参数搜索纳入最终性能估计，必须把超参数搜索也放入每个外层训练集的内层流程。",
        "结局为原始日级athlete-submitted injury notification，不等同于临床确诊损伤或新发损伤。",
        "",
        "依据：",
        "- https://www.bmj.com/content/385/bmj-2023-078378",
        "- https://www.bmj.com/content/388/bmj-2024-082505",
        "- https://jmlr.csail.mit.edu/papers/v11/cawley10a.html",
        "- https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html",
        "- https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432",
        "- https://imbalanced-learn.org/stable/common_pitfalls.html",
    ]
    (OUT / "NESTED_JOINT_REPORT.md").write_text("\n".join(report))

    manifest = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "source_output": str(SOURCE_OUT),
        "input_history_sha256": {str(w): sha256(SOURCE_OUT / f"history_{w}d.parquet") for w in WINDOWS},
        "outer_folds": 50,
        "inner_folds": 3,
        "candidate_count": len(candidates),
        "candidate_rule": "8 models x 6 windows x natural-prevalence control; fixed baseline parameters",
        "selection_metric": "mean inner AP, ROC-AUC tie-breaker",
        "target": "injury_report_t copied to injury; no label shift",
        "prediction_history": "calendar interval [date-window, date-1]",
        "literature_basis": [
            "TRIPOD+AI 2024",
            "PROBAST+AI 2025",
            "Cawley and Talbot 2010",
            "Saito and Rehmsmeier 2015",
        ],
    }
    (OUT / "NESTED_RUN_MANIFEST.json").write_text(json.dumps(json_safe(manifest), ensure_ascii=False, indent=2))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    candidates = make_candidates()
    reference = pd.read_parquet(SOURCE_OUT / "history_32d.parquet")
    players = sorted(reference.player_name.unique())
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = [(i, player, candidates) for i, player in enumerate(players)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_outer_fold, *task) for task in tasks]
        for future in as_completed(futures):
            fold = future.result()
            print(f"[03r] outer LOPO fold {fold + 1}/50 complete", flush=True)
    summarize(candidates)
    print((OUT / "NESTED_JOINT_REPORT.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
