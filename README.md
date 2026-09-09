# Football Predictor 2026

A production-oriented Python pipeline for supporting **2026/27** club-football analysis. It trains only on `2021-2022` through `2025-2026` across the Premier League, La Liga, Serie A, Bundesliga, Ligue 1, Champions League, Europa League and Conference League.

## What it does

- Downloads/caches FBref schedules and Opta-style team/player tables through `soccerdata.FBref`.
- Builds chronological, leakage-safe Elo, all-competition rolling points/goal difference, rest-day and neutral-venue match features.
- Fits a season-held-out multinomial logistic-regression outcome model and writes `models/match_outcome.joblib` plus accuracy/log loss/Brier/confusion-matrix metrics.
- Reads FBref event logs where available, derives 105m × 68m shot geometry, flags deterministic coordinate imputation, and fits an XGBoost goal model saved as `models/shot_xg.joblib`.
- Writes match performance, shot ROC/calibration, top-player heatmaps, and top-team goals-vs-model-xG dashboards to `outputs/`.

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --cache-dir .soccerdata-cache --output-dir outputs
pytest -q
```

`soccerdata` API coverage is source and competition dependent: schedules and season tables are useful broadly, while shot event logs may not be available for every requested competition. The runner completes match training and reports that limitation rather than silently dropping seasons. Cache the raw source responses for reproducible work.

## Data and missingness policy

- Allowed competition and season lists are explicit in `src/data_ingestion.py`; input is filtered to these training-only values.
- Pre-match rolling windows use whatever historical matches exist (up to five), and newly seen clubs start at Elo 1500.
- Numeric model inputs use fitted median imputation; categorical missingness becomes `unknown`.
- Missing shot coordinates are deterministically imputed to a conservative shooting region and marked `coords_imputed`; absent pressure becomes zero with `pressure_imputed=1`. Source xG is never the label.
- FBref shooting, passing, possession and defense tables are loaded separately so goals/xG, shots/SoT, conversion, passing, crossing and defensive measures are available for enrichment and interpretation. Join source-specific fields to your canonical team/player IDs before adding them to production feature sets.

## Layout

```text
src/data_ingestion.py       FBref/soccerdata ingestion
src/feature_engineering.py  Elo, rolling and shot geometry features
src/match_model.py          chronological logistic-regression pipeline
src/shot_model.py           chronological XGBoost pipeline
src/visualization.py        diagnostics, heatmaps and dashboards
main.py                     batch orchestration
```
