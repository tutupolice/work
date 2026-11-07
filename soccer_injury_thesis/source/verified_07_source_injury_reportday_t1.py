#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the SoccerMon athlete-day injury-report outcome source table.

Scope is deliberately limited to raw-row auditing, exact-row deduplication,
descriptive body-location/severity summaries, and one binary report-day row
per player and calendar date. No across-day reconstruction is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data/subjective/injury/injury.csv"
DEFAULT_OUTPUT = ROOT / "verified_outputs/07_source_injury_reportday_t1_v1"
SCHEMA_VERSION = "1.0.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def parse_payload(value: Any, raw_row_number: int) -> dict[str, str]:
    try:
        payload = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Invalid injury JSON at raw row {raw_row_number}: {value!r}"
        ) from exc
    if not isinstance(payload, dict) or not payload:
        raise ValueError(
            f"Expected a non-empty body-location dictionary at raw row "
            f"{raw_row_number}: {payload!r}"
        )
    return {
        str(location).strip(): str(severity).strip().lower()
        for location, severity in payload.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    args.outdir.mkdir(parents=True, exist_ok=True)

    raw_source = pd.read_csv(args.input)
    required = {"player_name", "type", "timestamp"}
    missing = required - set(raw_source.columns)
    if missing:
        raise ValueError(f"injury.csv missing columns: {sorted(missing)}")

    original_columns = list(raw_source.columns)
    raw = raw_source.copy()
    raw.insert(0, "raw_row_number", np.arange(1, len(raw) + 1, dtype=int))
    raw["date"] = pd.to_datetime(
        raw["timestamp"], format="%d.%m.%Y", errors="coerce"
    ).dt.normalize()
    if raw["date"].isna().any():
        invalid = raw.loc[raw["date"].isna()].copy()
        invalid.to_csv(args.outdir / "00_invalid_dates.csv", index=False)
        raise ValueError("Invalid DD.MM.YYYY timestamps; see 00_invalid_dates.csv")

    # Exact duplicates are defined over every original raw CSV column.
    duplicate_group_size = raw.groupby(original_columns, dropna=False)[
        "raw_row_number"
    ].transform("size")
    duplicate_removed = raw.duplicated(subset=original_columns, keep="first")
    duplicate_involved = duplicate_group_size.gt(1)
    raw["exact_duplicate_group_size"] = duplicate_group_size.astype(int)
    raw["is_exact_duplicate_removed"] = duplicate_removed.astype(int)

    duplicate_audit = raw.loc[duplicate_involved].copy()
    duplicate_audit["duplicate_audit_status"] = np.where(
        duplicate_audit["is_exact_duplicate_removed"].eq(1),
        "removed_exact_duplicate",
        "retained_first_exact_row",
    )

    clean = raw.loc[~duplicate_removed].copy().reset_index(drop=True)
    clean.insert(0, "clean_row_number", np.arange(1, len(clean) + 1, dtype=int))

    location_rows: list[dict[str, Any]] = []
    for row in clean.itertuples(index=False):
        payload = parse_payload(row.type, int(row.raw_row_number))
        for location, severity in payload.items():
            location_rows.append(
                {
                    "raw_row_number": int(row.raw_row_number),
                    "player_name": str(row.player_name),
                    "date": row.date,
                    "body_location": location,
                    "reported_severity": severity,
                }
            )
    locations = pd.DataFrame(location_rows)
    if locations.empty:
        raise ValueError("No body-location entries were produced from injury.csv")

    report_day = (
        clean[["player_name", "date"]]
        .drop_duplicates()
        .sort_values(["player_name", "date"])
        .reset_index(drop=True)
    )
    report_day["injury_report"] = np.int8(1)

    report_descriptive = (
        locations.groupby(["player_name", "date"], as_index=False)
        .agg(
            reported_location_entries=("body_location", "size"),
            unique_reported_locations=("body_location", "nunique"),
            minor_location_entries=(
                "reported_severity", lambda values: int((values == "minor").sum())
            ),
            major_location_entries=(
                "reported_severity", lambda values: int((values == "major").sum())
            ),
        )
    )
    report_descriptive["multi_location_report_day"] = (
        report_descriptive["unique_reported_locations"] > 1
    ).astype(int)

    player_summary = (
        report_day.groupby("player_name", as_index=False)
        .agg(
            injury_report_days=("date", "nunique"),
            first_report_date=("date", "min"),
            last_report_date=("date", "max"),
        )
        .merge(
            report_descriptive.groupby("player_name", as_index=False).agg(
                location_entries=("reported_location_entries", "sum"),
                minor_location_entries=("minor_location_entries", "sum"),
                major_location_entries=("major_location_entries", "sum"),
                multi_location_report_days=("multi_location_report_day", "sum"),
            ),
            on="player_name",
            how="left",
            validate="one_to_one",
        )
        .sort_values(["injury_report_days", "player_name"], ascending=[False, True])
        .reset_index(drop=True)
    )

    location_report_days = (
        locations[["body_location", "player_name", "date"]]
        .drop_duplicates()
        .groupby("body_location", as_index=False)
        .size()
        .rename(columns={"size": "player_report_days"})
    )
    location_summary = (
        locations.groupby("body_location", as_index=False)
        .agg(
            location_entries=("body_location", "size"),
            players_with_location=("player_name", "nunique"),
            minor_entries=(
                "reported_severity", lambda values: int((values == "minor").sum())
            ),
            major_entries=(
                "reported_severity", lambda values: int((values == "major").sum())
            ),
        )
        .merge(location_report_days, on="body_location", how="left", validate="one_to_one")
        .sort_values(["location_entries", "body_location"], ascending=[False, True])
        .reset_index(drop=True)
    )

    # Required invariants.
    if len(raw) != len(clean) + int(duplicate_removed.sum()):
        raise AssertionError("Raw/clean/duplicate row accounting failed")
    if clean.duplicated(subset=original_columns).any():
        raise AssertionError("Exact duplicate raw rows remain after cleaning")
    if report_day.duplicated(["player_name", "date"]).any():
        raise AssertionError("Duplicate player-date keys remain in report-day table")
    if not report_day["injury_report"].eq(1).all():
        raise AssertionError("Report-day table contains a label other than 1")
    if clean[["player_name", "date"]].drop_duplicates().shape[0] != len(report_day):
        raise AssertionError("Clean raw rows do not map one-to-one to report days")
    forbidden_label_columns = {
        "body_location", "reported_severity", "minor", "major",
        "episode", "ongoing_injury", "reinjury", "recovery",
    }
    if forbidden_label_columns.intersection(report_day.columns):
        raise AssertionError("Descriptive injury fields leaked into daily label table")

    severity_counts = locations["reported_severity"].value_counts(dropna=False)
    raw_audit = pd.DataFrame(
        [
            ("raw_rows", len(raw)),
            ("exact_duplicate_rows_involved", int(duplicate_involved.sum())),
            ("exact_duplicates_removed", int(duplicate_removed.sum())),
            ("clean_raw_rows", len(clean)),
            ("unique_player_report_days", len(report_day)),
            ("players_with_reports", report_day["player_name"].nunique()),
            ("date_min", report_day["date"].min().strftime("%Y-%m-%d")),
            ("date_max", report_day["date"].max().strftime("%Y-%m-%d")),
            ("body_location_entries", len(locations)),
            ("unique_body_locations", locations["body_location"].nunique()),
            ("minor_location_entries", int(severity_counts.get("minor", 0))),
            ("major_location_entries", int(severity_counts.get("major", 0))),
            (
                "other_severity_entries",
                int(len(locations) - severity_counts.get("minor", 0) - severity_counts.get("major", 0)),
            ),
            (
                "multi_location_report_days",
                int(report_descriptive["multi_location_report_day"].sum()),
            ),
        ],
        columns=["metric", "value"],
    )

    raw_audit.to_csv(args.outdir / "01_injury_raw_audit.csv", index=False)
    duplicate_audit.to_csv(args.outdir / "02_injury_duplicates.csv", index=False)
    clean.to_csv(args.outdir / "03_injury_raw_clean.csv", index=False)
    report_day.to_csv(args.outdir / "04_injury_report_day.csv", index=False)
    player_summary.to_csv(args.outdir / "05_injury_player_summary.csv", index=False)
    location_summary.to_csv(args.outdir / "06_injury_location_summary.csv", index=False)

    report_lines = [
        "# SoccerMon injury report-day audit",
        "",
        "本阶段只执行完全重复行清理和player × date日级injury_report构建。未执行跨日期合并或任何episode相关处理。",
        "",
        "## Raw audit",
        "",
        "```text",
        raw_audit.to_string(index=False),
        "```",
        "",
        "## Reports per player",
        "",
        "```text",
        player_summary.to_string(index=False),
        "```",
        "",
        "## Location and severity distribution",
        "",
        "```text",
        location_summary.to_string(index=False),
        "```",
        "",
        "`04_injury_report_day.csv`仅包含player_name、date和injury_report；location与severity仅保留在描述性审计文件中。",
    ]
    (args.outdir / "INJURY_REPORTDAY_REPORT.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now().astimezone(),
        "script": Path(__file__).name,
        "input_path": args.input,
        "input_sha256": sha256_file(args.input),
        "timestamp_format": "%d.%m.%Y",
        "analysis_unit": "player-calendar-day",
        "daily_label": "injury_report",
        "daily_label_definition": (
            "1 when at least one cleaned raw injury report exists for player_name and date"
        ),
        "across_day_merging": False,
        "location_or_severity_in_daily_label_file": False,
        "counts": dict(zip(raw_audit["metric"], raw_audit["value"])),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    (args.outdir / "manifest.json").write_text(
        json.dumps(native(manifest), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("INJURY REPORT-DAY BUILD COMPLETED")
    print(f"Input:  {args.input}")
    print(f"Output: {args.outdir}")
    print(raw_audit.to_string(index=False))


if __name__ == "__main__":
    main()
