"""Plots kept separate from model training to support notebooks and batch runs."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import RocCurveDisplay

def _save(path: str|Path): Path(path).parent.mkdir(parents=True,exist_ok=True); plt.tight_layout(); plt.savefig(path,dpi=150); plt.close()
def plot_match_performance(predictions: pd.DataFrame, path: str|Path):
    """Save accuracy grouped by available competition field."""
    group=next((c for c in ["competition","league","season"] if c in predictions),"season"); acc=predictions.assign(correct=predictions.result.eq(predictions.prediction)).groupby(group).correct.mean().sort_values()
    acc.plot.bar(color="#1f77b4"); plt.ylabel("Accuracy"); plt.ylim(0,1); plt.title("Match outcome accuracy"); _save(path)
def plot_shot_diagnostics(shots: pd.DataFrame, target: str, path: str|Path):
    """Plot ROC and calibration for held-out shots."""
    fig,ax=plt.subplots(1,2,figsize=(10,4)); RocCurveDisplay.from_predictions(shots[target],shots.predicted_xg,ax=ax[0]); obs,pred=calibration_curve(shots[target],shots.predicted_xg,n_bins=10); ax[1].plot(pred,obs,"o-"); ax[1].plot([0,1],[0,1],"--"); ax[1].set(title="Calibration",xlabel="Predicted",ylabel="Observed"); _save(path)
def player_heatmap(shots: pd.DataFrame, player: str, path: str|Path):
    """Render an average predicted-xG heatmap on a 105 x 68m pitch."""
    name=next(c for c in shots if c.lower() in {"player","player_name"}); d=shots[shots[name].eq(player)]; fig,ax=plt.subplots(figsize=(9,5)); ax.hist2d(d.x,d.y,bins=(12,8),weights=d.predicted_xg,density=False,cmap="magma"); ax.set(xlim=(0,105),ylim=(0,68),title=f"{player}: predicted goal probability",xlabel="Pitch length (m)",ylabel="Pitch width (m)"); _save(path)
def team_dashboard(shots: pd.DataFrame, team: str, path: str|Path):
    """Save a compact player goals vs model-xG dashboard for one club."""
    tcol=next(c for c in shots if c.lower() in {"team","squad"}); pcol=next(c for c in shots if c.lower() in {"player","player_name"}); goal=next(c for c in shots if c.lower() in {"goal","is_goal","outcome_goal"}); d=shots[shots[tcol].eq(team)].groupby(pcol).agg(goals=(goal,"sum"),model_xg=("predicted_xg","sum")).sort_values("goals",ascending=False).head(10); d.plot.barh(figsize=(8,5)); plt.title(f"{team}: goals vs model xG"); _save(path)
