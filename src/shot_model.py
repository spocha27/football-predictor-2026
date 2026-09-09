"""Shot-level XGBoost goal probability model."""
from __future__ import annotations
from pathlib import Path
import joblib, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

NUMERIC=["x","y","distance_m","angle_rad","defender_pressure","coords_imputed","pressure_imputed"]
CATEGORICAL=["body_part","assist_type","situation","location_zone"]
def train_shot_model(shots: pd.DataFrame, test_season: str="2025-2026", model_path: str|Path="models/shot_xg.joblib") -> tuple[Pipeline,dict,pd.DataFrame]:
    """Train a calibrated-input gradient booster, retaining xG as validation—not target—data."""
    target=next((c for c in shots if c.lower() in {"goal","is_goal","outcome_goal"}),None)
    if target is None: raise ValueError("Shot data must contain a goal/is_goal target column.")
    split=shots.season.astype(str).eq(test_season); train,test=shots.loc[~split],shots.loc[split]
    if train.empty or test.empty or train[target].nunique()<2: raise ValueError("Need both seasons and both goal outcomes in training.")
    nums=[c for c in NUMERIC if c in shots]; cats=[c for c in CATEGORICAL if c in shots]
    prep=ColumnTransformer([("num",SimpleImputer(strategy="median",add_indicator=True),nums),("cat",Pipeline([("impute",SimpleImputer(strategy="constant",fill_value="unknown")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cats)])
    model=Pipeline([("prep",prep),("classifier",XGBClassifier(n_estimators=300,max_depth=3,learning_rate=.04,subsample=.85,colsample_bytree=.85,eval_metric="logloss",random_state=42,n_jobs=1))]); model.fit(train,train[target].astype(int))
    p=model.predict_proba(test)[:,1]; metrics={"log_loss":log_loss(test[target],p),"auc":roc_auc_score(test[target],p)}
    out=test.copy(); out["predicted_xg"]=p; Path(model_path).parent.mkdir(parents=True,exist_ok=True); joblib.dump(model,model_path)
    return model,metrics,out
