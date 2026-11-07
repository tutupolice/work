"""Score a supplied, aligned 75-feature CSV with the frozen source pipeline."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new output path; existing results are not overwritten.")
    frozen = Path(__file__).resolve().parents[1] / "source/outputs/injury_history_lopo_v1/frozen_source_candidate"
    features = json.loads((frozen / "SELECTED_FEATURES.json").read_text())["features"]
    frame = pd.read_csv(args.input)
    missing = set(features) - set(frame)
    if missing:
        raise ValueError(f"Input is not aligned; missing {len(missing)} required columns")
    model = joblib.load(frozen / "SOCCERMON_INJURY_8D_LIGHTGBM_PIPELINE.joblib")
    pred = model.predict_proba(frame[features].astype(float))[:, 1]
    if not np.isfinite(pred).all():
        raise ValueError("Non-finite model output")
    result = frame[[c for c in ("player_name", "date", "injury") if c in frame]].copy()
    result["source_raw_probability"] = pred
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Scored {len(result)} rows; output contains participant-level information.")

if __name__ == "__main__":
    main()
