"""Run the five-season training pipeline (network/cache availability required)."""
from __future__ import annotations
import argparse
from pathlib import Path
from src.data_ingestion import load_matches, load_shots
from src.feature_engineering import build_match_features, build_shot_features
from src.match_model import train_match_model
from src.shot_model import train_shot_model
from src.visualization import plot_match_performance, plot_shot_diagnostics, player_heatmap, team_dashboard

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--cache-dir",default=None); parser.add_argument("--output-dir",default="outputs"); args=parser.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    matches=build_match_features(load_matches(cache_dir=args.cache_dir)); _,match_metrics,match_predictions=train_match_model(matches); print("Match metrics:",match_metrics); plot_match_performance(match_predictions,out/"match_performance.png")
    raw_shots=load_shots(cache_dir=args.cache_dir)
    if raw_shots.empty:
        print("No shot event log returned by FBref; match model completed. Use a supported event source/cache for shot training."); return
    shots=build_shot_features(raw_shots); _,shot_metrics,scored=train_shot_model(shots); print("Shot metrics:",shot_metrics)
    target=next(c for c in scored if c.lower() in {"goal","is_goal","outcome_goal"}); plot_shot_diagnostics(scored,target,out/"shot_diagnostics.png")
    player_col=next((c for c in scored if c.lower() in {"player","player_name"}),None); team_col=next((c for c in scored if c.lower() in {"team","squad"}),None)
    if player_col:
        for player in scored.groupby(player_col).predicted_xg.sum().nlargest(3).index: player_heatmap(scored,player,out/f"heatmap_{str(player).replace('/','_')}.png")
    if team_col:
        for team in scored.groupby(team_col).predicted_xg.sum().nlargest(3).index: team_dashboard(scored,team,out/f"dashboard_{str(team).replace('/','_')}.png")
if __name__ == "__main__": main()
