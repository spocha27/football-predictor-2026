"""FBref/soccerdata ingestion with an intentionally small, stable output schema."""
from __future__ import annotations
from pathlib import Path
from typing import Literal
import pandas as pd

COMPETITIONS = ["ENG-Premier League", "ESP-La Liga", "ITA-Serie A", "GER-Bundesliga", "FRA-Ligue 1", "UEFA-Champions League", "UEFA-Europa League", "UEFA-Europa Conference League"]
SEASONS = ["2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"]


def _fbref(leagues: list[str], seasons: list[str], cache_dir: str | Path | None = None):
    """Create FBref reader once; soccerdata handles caching and HTTP requests."""
    import soccerdata as sd
    sd.configure_logging(to_console=True, to_file=False)
    kwargs = {"leagues": leagues, "seasons": seasons}
    if cache_dir: kwargs["data_dir"] = Path(cache_dir)
    return sd.FBref(**kwargs)


def _flat(frame: pd.DataFrame) -> pd.DataFrame:
    """Flatten soccerdata's MultiIndex columns without discarding source fields."""
    out = frame.reset_index()
    out.columns = ["_".join(str(v) for v in c if str(v) != "").strip("_") if isinstance(c, tuple) else str(c) for c in out.columns]
    return out


def load_matches(leagues: list[str] = COMPETITIONS, seasons: list[str] = SEASONS, cache_dir: str | Path | None = None) -> pd.DataFrame:
    """Download schedules/results; retain unfinished rows for future 2026/27 inference only."""
    data = _flat(_fbref(leagues, seasons, cache_dir).read_schedule())
    return data[data.get("league", data.get("league_id", "")).isin(leagues) & data.get("season", "").astype(str).isin(seasons)].copy()


def load_team_metrics(leagues: list[str] = COMPETITIONS, seasons: list[str] = SEASONS, cache_dir: str | Path | None = None) -> pd.DataFrame:
    """Load Opta-style FBref season tables (shooting/passing/possession) and tag stat source."""
    reader = _fbref(leagues, seasons, cache_dir); tables = []
    for stat_type in ("standard", "shooting", "passing", "possession", "defense"):
        try:
            table = _flat(reader.read_team_season_stats(stat_type=stat_type)); table["stat_type"] = stat_type; tables.append(table)
        except Exception as exc: print(f"Skipping team {stat_type}: {exc}")
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def load_player_metrics(leagues: list[str] = COMPETITIONS, seasons: list[str] = SEASONS, cache_dir: str | Path | None = None) -> pd.DataFrame:
    """Load player tables used to enrich dashboard summaries."""
    reader = _fbref(leagues, seasons, cache_dir); tables = []
    for stat_type in ("standard", "shooting", "passing", "defense", "possession"):
        try:
            table = _flat(reader.read_player_season_stats(stat_type=stat_type)); table["stat_type"] = stat_type; tables.append(table)
        except Exception as exc: print(f"Skipping player {stat_type}: {exc}")
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def load_shots(leagues: list[str] = COMPETITIONS, seasons: list[str] = SEASONS, cache_dir: str | Path | None = None) -> pd.DataFrame:
    """Retrieve FBref events, then keep shot events when detailed event logs are available.

    FBref coverage differs by competition; callers should persist this raw table and inspect
    its schema. Missing event fields are handled in feature_engineering rather than dropping a season.
    """
    reader = _fbref(leagues, seasons, cache_dir)
    try:
        events = _flat(reader.read_events())
    except Exception as exc:
        print(f"Shot events unavailable from FBref: {exc}"); return pd.DataFrame()
    type_col = next((c for c in events.columns if c.lower() in {"type", "event_type", "type_name"}), None)
    return events[events[type_col].astype(str).str.contains("shot", case=False, na=False)].copy() if type_col else events
