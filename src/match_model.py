"""Multiclass, season-held-out match outcome model."""
from __future__ import annotations
from pathlib import Path
import joblib, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = ["elo_difference", "home_ppg_5", "away_ppg_5", "home_gd_5", "away_gd_5", "home_rest_days", "away_rest_days", "home_advantage", "is_neutral"]
def train_match_model(features: pd.DataFrame, test_season: str = "2025-2026", model_path: str | Path = "models/match_outcome.joblib") -> tuple[Pipeline, dict, pd.DataFrame]:
    """Fit only earlier seasons and evaluate a chronological 2025/26 holdout."""
    available = [x for x in FEATURES if x in features]; cat = [x for x in ["league", "competition"] if x in features]
    split = features["season"].astype(str).eq(test_season); train, test = features.loc[~split], features.loc[split]
    if train.empty or test.empty: raise ValueError("Need both pre-2025/26 and 2025/26 completed matches.")
    prep = ColumnTransformer([("num", Pipeline([("impute",SimpleImputer(strategy="median", add_indicator=True)),("scale",StandardScaler())]),available), ("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat)])
    model = Pipeline([("prep",prep),("classifier",LogisticRegression(max_iter=3000))]); model.fit(train,train.result)
    prob=model.predict_proba(test); pred=model.predict(test); labels=list(model.classes_)
    metrics={"accuracy":accuracy_score(test.result,pred),"log_loss":log_loss(test.result,prob,labels=labels),"confusion_matrix":confusion_matrix(test.result,pred,labels=labels).tolist()}
    # Multi-class Brier score, averaged over one-vs-rest outcomes.
    metrics["brier_score"] = sum(brier_score_loss((test.result==label).astype(int),prob[:,i]) for i,label in enumerate(labels))/len(labels)
    out=test[[c for c in ["season","league","competition","result"] if c in test]].copy(); out["prediction"]=pred
    for i,label in enumerate(labels): out[f"prob_{label}"]=prob[:,i]
    Path(model_path).parent.mkdir(parents=True,exist_ok=True); joblib.dump(model,model_path)
    return model,metrics,out
