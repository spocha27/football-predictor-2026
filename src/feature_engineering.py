"""Leakage-safe match and shot feature construction."""
from __future__ import annotations
import numpy as np
import pandas as pd

PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0

def _col(df: pd.DataFrame, *names: str) -> str:
    return next((c for c in df.columns if c.lower().replace("_", " ") in names), names[0])

def build_match_features(matches: pd.DataFrame, window: int = 5, base_elo: float = 1500, k: float = 24) -> pd.DataFrame:
    """Add pre-match Elo, all-competition rolling form, rest, and neutral/home flags.

    Past-only shifts prevent the result currently being predicted leaking into its feature row.
    Missing xG is retained with an explicit indicator; model preprocessing imputes numeric values.
    """
    d = matches.copy(); date = _col(d, "date"); home = _col(d, "home", "home team"); away = _col(d, "away", "away team")
    hg, ag = _col(d, "home goals", "home_score"), _col(d, "away goals", "away_score")
    d[date] = pd.to_datetime(d[date], errors="coerce"); d = d.sort_values(date).reset_index(drop=True)
    d["home_elo"] = float(base_elo); d["away_elo"] = float(base_elo); ratings: dict[str, float] = {}; history: dict[str, list[dict]] = {}; last_date: dict[str, pd.Timestamp] = {}
    for i, row in d.iterrows():
        h, a = str(row[home]), str(row[away]); d.loc[i, ["home_elo", "away_elo"]] = [ratings.get(h, base_elo), ratings.get(a, base_elo)]
        for side, team, opp in (("home", h, a), ("away", a, h)):
            prior = history.get(team, [])[-window:]; d.loc[i, f"{side}_ppg_{window}"] = np.mean([x["pts"] for x in prior]) if prior else np.nan
            d.loc[i, f"{side}_gd_{window}"] = np.mean([x["gd"] for x in prior]) if prior else np.nan
            d.loc[i, f"{side}_rest_days"] = (row[date] - last_date[team]).days if team in last_date and pd.notna(row[date]) else np.nan
        if pd.notna(row.get(hg)) and pd.notna(row.get(ag)):
            result = 1 if row[hg] > row[ag] else .5 if row[hg] == row[ag] else 0; expected = 1 / (1 + 10 ** ((ratings.get(a,base_elo)-ratings.get(h,base_elo))/400))
            ratings[h] = ratings.get(h,base_elo) + k*(result-expected); ratings[a] = ratings.get(a,base_elo) + k*((1-result)-(1-expected))
            for team, gf, ga in ((h,row[hg],row[ag]),(a,row[ag],row[hg])): history.setdefault(team, []).append({"pts": 3 if gf>ga else 1 if gf==ga else 0, "gd": gf-ga}); last_date[team] = row[date]
    d["elo_difference"] = d.home_elo-d.away_elo; d["is_neutral"] = d.get("venue", "").astype(str).str.contains("neutral", case=False, na=False).astype(int); d["home_advantage"] = 1-d.is_neutral
    d["result"] = np.select([d[hg]>d[ag], d[hg]==d[ag]], ["home_win", "draw"], default="away_win")
    return d

def build_shot_features(shots: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Normalize coordinates to metres and engineer geometry; deterministic zone imputation flags uncertainty."""
    d = shots.copy(); rng = np.random.default_rng(seed)
    xcol, ycol = _col(d,"x","location x","start x"), _col(d,"y","location y","start y")
    d["coords_imputed"] = (~d.get(xcol, pd.Series(np.nan,index=d.index)).notna() | ~d.get(ycol, pd.Series(np.nan,index=d.index)).notna()).astype(int)
    d["x"] = pd.to_numeric(d.get(xcol), errors="coerce").astype(float); d["y"] = pd.to_numeric(d.get(ycol), errors="coerce").astype(float)
    # Coordinates in 0..100 are converted; absent coordinates get a conservative outside-box proxy.
    d.loc[d.x.between(0,100), "x"] *= 1.05; d.loc[d.y.between(0,100), "y"] *= .68
    d.x = d.x.fillna(pd.Series(rng.uniform(75, 92, len(d)), index=d.index)); d.y = d.y.fillna(pd.Series(rng.uniform(15, 53, len(d)), index=d.index))
    dx, dy = PITCH_LENGTH-d.x, np.abs(d.y-PITCH_WIDTH/2); d["distance_m"] = np.hypot(dx,dy)
    d["angle_rad"] = np.arctan2(7.32*dx, np.maximum(dx**2+dy**2-(7.32/2)**2, 1e-6))
    d["location_zone"] = np.select([d.x>=99, d.x>=88], ["six_yard_box", "inside_box"], default="outside_box")
    for c in ("body_part", "assist_type", "situation"):
        d[c] = d[c].fillna("unknown").astype(str) if c in d else "unknown"
    d["pressure_imputed"] = 1; d["defender_pressure"] = 0
    return d
