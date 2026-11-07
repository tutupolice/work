# SoccerMon source athlete-day preprocessing report

本程序完成原始来源键联合、日级数据清洗、时间格式统一和主客观数据合并。
未进行缺失值插补、标准化、窗口特征、特征选择、EDA或模型训练。

- raw source-key union rows: 21,729
- raw source-key union players: 78
- raw injury-only rows retained: 0
- raw illness-only rows retained: 0
- output rows: 19,689
- players: 50
- date range: 2020-01-09 to 2021-12-31
- y_t1 positives among non-boundary rows: 137
- outcome-evaluable rows: 19,639
- outcome boundary rows: 50
- GPS record coverage: 40.40%
- duplicate player_name + date keys: 0

## Input sources

- training-load/session.json: session-level records aggregated to player-day.
- training-load/*.csv: daily/rolling source-native exports retained as fields; they do not create prediction origins.
- wellness/*.csv: all seven source wellness files reshaped from wide to long.
- game-performance/game-performance.csv and illness/illness.csv: date-normalized daily reports.
- verified_outputs/04_source_gps/source_gps_daily.csv: verified daily GPS cache; raw Parquet was not read.
- verified_outputs/07_source_injury_reportday_t1_v1/04_injury_report_day.csv: locked outcome/QC only.

## Integrity decisions

- Join key: player_name + date.
- Raw union: session, wellness, game, illness, GPS, and locked injury report-day keys are retained for audit.
- Cohort eligibility: actual PmSys report presence by player_name + calendar_year.
- Prediction origins: session, wellness, game-performance, or verified GPS presence.
- Injury-only and illness-only dates are retained in the raw union but are not primary prediction origins.
- GPS missing values remain NA; no unconditional zero filling was performed.
- Session daily sums use min_count=1; an existing session day with all numeric values missing remains NA.
- injury_report_t and y_t1 are outcome/QC fields and are forbidden predictors.
- No rolling windows were created in this stage.
- No imputation, standardization, feature selection, EDA, or model fitting was performed.

## Output

- source_athlete_day_raw_union.csv: written
- source_athlete_day_preprocessed.csv: written
- source_athlete_day_raw_union.parquet: written
- source_athlete_day_preprocessed.parquet: written