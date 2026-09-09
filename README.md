# Football Predictor 2026

Predict match outcomes and shot-level goal probabilities for Europe’s top clubs (2026/27 season), trained on the last 5 seasons (2021/22–2025/26) from:

- Premier League, La Liga, Serie A, Bundesliga, Ligue 1  
- UEFA Champions League, Europa League, Conference League  

## Features

- **Match outcome model**: Logistic regression using Elo ratings, recent form, goals/xG, home advantage, etc.
- **Shot-level goal model**: XGBoost classifier using shot location, angle, distance, body part, assist type, game situation, defender pressure proxies.
- **Dashboards**:
  - Team pages showing top scorers, predicted xG, over/underperformance.
  - Player heatmaps showing where they are most likely to score.

## Tech stack

- Python 3.10+
- `pandas`, `scikit-learn`, `xgboost`
- `soccerdata` (FBref/Opta-style data)
- `matplotlib`, `seaborn`
- Optional: `pytest`, `ipykernel`

## Project structure

```text
.
├─ src/
│  ├─ data_ingestion.py
│  ├─ feature_engineering.py
│  ├─ match_model.py
│  ├─ shot_model.py
│  ├─ visualization.py
│  └─ utils.py
├─ notebooks/
├─ tests/
├─ models/
├─ outputs/
├─ main.py
├─ requirements.txt
└─ README.md
```

## Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/football-predictor-2026.git
   cd football-predictor-2026
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the main pipeline:
   ```bash
   python main.py
   ```

This will:
- Download data for the last 5 seasons from supported competitions.
- Train match-outcome and shot-level models.
- Save models to `models/`.
- Generate example plots and dashboards in `outputs/`.

## Usage examples

(To be added: example notebooks, how to generate team dashboards, etc.)

## License

MIT (or your preferred license).
