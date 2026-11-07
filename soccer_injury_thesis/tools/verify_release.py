"""Read-only checks for the public research package; no model fitting."""
from pathlib import Path
import argparse
import ast
import csv
import hashlib
import json
import tarfile
from checkpoint_archive import open_archive

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    provenance = json.loads((ROOT / "BUILD_PROVENANCE.json").read_text())
    assert provenance["local_participant_data_included"] is False
    for rec in provenance["files"]:
        if "segments" in rec:
            digest = hashlib.sha256()
            for part in rec["segments"]:
                data = (ROOT / part["path"]).read_bytes()
                assert hashlib.sha256(data).hexdigest() == part["sha256"]
                digest.update(data)
            assert digest.hexdigest() == rec["packaged_sha256"]
        else:
            assert sha(ROOT / rec["destination"]) == rec["packaged_sha256"], rec["destination"]
    scripts = sorted(ROOT.rglob("*.py"))
    for path in scripts:
        ast.parse(path.read_text(), filename=str(path))
    local_files = [p for p in (ROOT / "local").rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    assert all(p.suffix == ".py" or p.name == "README.md" for p in local_files)
    source = ROOT / "source"
    table = source / "verified_outputs/01_source_preprocessing_v1/source_athlete_day_preprocessed.csv"
    with table.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    assert len(rows) == 19689 and len(fields) == 65
    assert len({r["player_name"] for r in rows}) == 50
    assert len({(r["player_name"], r["date"]) for r in rows}) == len(rows)
    assert sum(int(float(r["injury_report_t"])) for r in rows) == 156
    manifest = json.loads((source / "cache/BENCHMARK_CACHE_MANIFEST.json").read_text())
    expected = {r["path"]: r for r in manifest}
    members = 0
    with open_archive(source) as tf:
        for member in tf:
            assert member.name in expected and member.isfile()
            data = tf.extractfile(member).read()
            assert len(data) == expected[member.name]["bytes"]
            assert hashlib.sha256(data).hexdigest() == expected[member.name]["sha256"]
            members += 1
    assert members == 4800 == len(expected)
    checks = {"provenance_records_verified": len(provenance["files"]),
              "python_files_parsed": len(scripts), "merged_source_rows": len(rows),
              "source_players": 50, "source_merged_positive_days": 156,
              "benchmark_fold_files_verified": members,
              "local_participant_data_published": False, "model_refitted_during_validation": False}
    if args.full:
        import importlib
        import sys
        import joblib
        import numpy as np
        import pandas as pd
        from sklearn.metrics import average_precision_score, roc_auc_score
        out = source / "outputs/injury_history_lopo_v1"
        frames = {w: pd.read_parquet(out / f"history_{w}d.parquet") for w in (2,4,8,16,21,32)}
        reference = frames[8][["player_name", "date", "injury"]]
        for frame in frames.values():
            assert frame.shape == (19613,78)
            assert int(frame.injury.sum()) == 149 and frame.player_name.nunique() == 50
            pd.testing.assert_frame_equal(reference, frame[["player_name", "date", "injury"]])
        a = pd.read_parquet(source / "verified_outputs/01_source_preprocessing_v1/source_athlete_day_preprocessed.parquet")
        b = pd.read_parquet(out / "merged_source_snapshot.parquet")
        pd.testing.assert_frame_equal(a, b)
        frozen = out / "frozen_source_candidate"
        features = json.loads((frozen / "SELECTED_FEATURES.json").read_text())["features"]
        assert len(features) == 75
        model = joblib.load(frozen / "SOCCERMON_INJURY_8D_LIGHTGBM_PIPELINE.joblib")
        scores = model.predict_proba(frames[8][features].iloc[:128].astype(float))[:, 1]
        assert np.isfinite(scores).all() and ((0 <= scores) & (scores <= 1)).all()
        assert set(np.asarray(model.classes_)) == {0,1}
        nested = pd.read_csv(source / "nested_48_reconciliation/nested_48_oof_predictions.csv")
        report = json.loads((source / "nested_48_reconciliation/RECONCILED_REPORT.json").read_text())
        assert len(nested) == 19613 and int(nested.injury.sum()) == 149
        assert not nested.duplicated(["player_name", "date"]).any()
        assert abs(average_precision_score(nested.injury,nested.score)-report["AP"]) < 1e-12
        assert abs(roc_auc_score(nested.injury,nested.score)-report["ROC_AUC"]) < 1e-12
        benchmark = pd.read_csv(out / "benchmark_summary.csv")
        assert len(benchmark) == 96 and benchmark.configuration.nunique() == 96
        sys.path.insert(0, str(source))
        m = importlib.import_module("03r_injury_nested_joint_selection")
        assert len(m.make_candidates()) == 48
        local = ROOT / "local/数据处理方案"
        sys.path.insert(0, str(local))
        for name in ("04_chapter4_local_target_audit", "05_target_domain_daily_rebuild",
                     "15_source_candidate_target_alignment", "16_target_domain_update_lopo"):
            importlib.import_module(name)
        checks.update({"history_windows_verified": 6, "history_rows": 19613,
                       "history_positive_days": 149, "features": 75,
                       "frozen_model_loaded_and_scored_rows": len(scores),
                       "nested_oof_AP_recomputed": report["AP"],
                       "nested_oof_ROC_AUC_recomputed": report["ROC_AUC"],
                       "benchmark_configurations": 96, "nested_control_candidates": 48,
                       "local_code_imports_checked": 4})
    print(json.dumps(checks, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
