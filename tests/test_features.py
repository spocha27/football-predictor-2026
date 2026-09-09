import pandas as pd
from src.feature_engineering import build_match_features, build_shot_features

def test_match_features_are_pre_match_and_have_elo():
    matches=pd.DataFrame({"date":["2021-08-01","2021-08-08"],"home":["A","B"],"away":["B","A"],"home goals":[2,0],"away goals":[0,1],"season":["2021-2022"]*2,"league":["ENG-Premier League"]*2,"venue":["Home","Home"]})
    got=build_match_features(matches)
    assert got.loc[0,"home_elo"] == 1500 and got.loc[1,"away_elo"] != 1500
    assert pd.isna(got.loc[0,"home_ppg_5"])
def test_shot_geometry_and_missing_coordinate_flags():
    got=build_shot_features(pd.DataFrame({"x":[90,None],"y":[50,None],"goal":[0,1]}))
    assert got.distance_m.notna().all() and got.coords_imputed.tolist()==[0,1]
